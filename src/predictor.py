"""
predictor.py - Real-time gesture inference with smoothing and confidence filtering
"""

import collections
import numpy as np
from typing import Optional, Tuple, List

from src.config import (
    CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW,
    UNKNOWN_LABEL, GESTURE_CLASSES, SEQUENCE_LENGTH,
    STATIC_MODEL_PATH, LSTM_MODEL_PATH, LABEL_ENCODER_PATH
)
from src.models import load_model
from src.dataset import DatasetLoader


class PredictionSmoother:
    """
    Temporal majority-vote smoother.
    Keeps a rolling window of (label, confidence) pairs and returns
    the most frequent label along with the mean confidence of that label.
    """

    def __init__(self, window: int = SMOOTHING_WINDOW):
        self._window  = window
        self._history: collections.deque = collections.deque(maxlen=window)

    def update(self, label: str, confidence: float) -> Tuple[str, float]:
        self._history.append((label, confidence))
        counter = collections.Counter(h[0] for h in self._history)
        smoothed_label = counter.most_common(1)[0][0]
        label_confs = [c for l, c in self._history if l == smoothed_label]
        smoothed_conf = float(np.mean(label_confs))
        return smoothed_label, smoothed_conf

    def reset(self):
        self._history.clear()


class StaticPredictor:
    """Single-frame landmark predictor."""

    def __init__(
        self,
        model_path:   str = STATIC_MODEL_PATH,
        encoder_path: str = LABEL_ENCODER_PATH,
        threshold:    float = CONFIDENCE_THRESHOLD,
        smoothing:    int   = SMOOTHING_WINDOW,
    ):
        self.model    = load_model(model_path)
        self.le       = DatasetLoader.load_label_encoder(encoder_path)
        self.threshold = threshold
        self.smoother  = PredictionSmoother(smoothing)

    def _infer(self, features: np.ndarray) -> np.ndarray:
        x = features.reshape(1, -1).astype(np.float32)
        return self.model(x, training=False).numpy()[0]

    def _resolve_label(self, probs: np.ndarray) -> Tuple[str, float]:
        """
        Resolve the top-1 label against the confidence threshold.

        A visually-confusable pair (e.g. U/V) can split softmax mass between
        the two classes so neither individually clears `self.threshold`, even
        though the model is confidently choosing between just those two. In
        that case fall back to accepting the argmax label if the combined
        mass of the top-2 candidates clears the threshold instead of forcing
        UNKNOWN_LABEL.
        """
        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf >= self.threshold:
            return self.le.classes_[idx], conf
        top2_idx = np.argsort(probs)[::-1][:2]
        top2_sum = float(probs[top2_idx[0]] + probs[top2_idx[1]])
        if top2_sum >= self.threshold:
            return self.le.classes_[idx], conf
        return UNKNOWN_LABEL, conf

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        features: shape (63,)
        Returns (label, confidence).
        label = UNKNOWN_LABEL if confidence < threshold.
        """
        probs = self._infer(features)
        label, conf = self._resolve_label(probs)
        return self.smoother.update(label, conf)

    def predict_top_k(
        self, features: np.ndarray, k: int = 3
    ) -> List[Tuple[str, float]]:
        """Return top-k (label, confidence) pairs."""
        probs = self._infer(features)
        top_k = np.argsort(probs)[::-1][:k]
        return [(self.le.classes_[i], float(probs[i])) for i in top_k]

    def predict_with_top_k(
        self, features: np.ndarray, k: int = 3
    ) -> Tuple[Tuple[str, float], List[Tuple[str, float]]]:
        """Single inference pass returning both smoothed top-1 and top-k list."""
        probs = self._infer(features)
        label, conf = self._resolve_label(probs)
        smoothed = self.smoother.update(label, conf)
        top_k_idx = np.argsort(probs)[::-1][:k]
        top_k = [(self.le.classes_[i], float(probs[i])) for i in top_k_idx]
        return smoothed, top_k


class LSTMPredictor:
    """Sequence-based LSTM predictor for dynamic gestures."""

    def __init__(
        self,
        model_path:   str = LSTM_MODEL_PATH,
        encoder_path: str = LABEL_ENCODER_PATH,
        seq_len:      int = SEQUENCE_LENGTH,
        threshold:    float = CONFIDENCE_THRESHOLD,
        smoothing:    int   = SMOOTHING_WINDOW,
    ):
        self.model    = load_model(model_path)
        self.le       = DatasetLoader.load_label_encoder(encoder_path)
        self.seq_len  = seq_len
        self.threshold = threshold
        self.smoother  = PredictionSmoother(smoothing)
        self._buffer: collections.deque = collections.deque(maxlen=seq_len)

    def push_frame(self, features: np.ndarray):
        """Add a single-frame feature vector to the rolling buffer."""
        self._buffer.append(features)

    def predict(self) -> Tuple[Optional[str], float]:
        """
        Predict from current buffer.
        Returns (None, 0.0) if buffer not full yet.
        """
        if len(self._buffer) < self.seq_len:
            return None, 0.0

        seq   = np.array(list(self._buffer), dtype=np.float32)  # (T, 63)
        x     = seq.reshape(1, self.seq_len, -1)
        probs = self.model(x, training=False).numpy()[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        label = self.le.classes_[idx] if conf >= self.threshold else UNKNOWN_LABEL
        return self.smoother.update(label, conf)

    def reset(self):
        self._buffer.clear()
        self.smoother.reset()


class EnsemblePredictor:
    """
    Combines static and LSTM predictions via weighted average.
    Falls back to static if LSTM buffer is not full yet.
    """

    def __init__(
        self,
        static_weight: float = 0.4,
        lstm_weight:   float = 0.6,
        threshold:     float = CONFIDENCE_THRESHOLD,
        smoothing:     int   = SMOOTHING_WINDOW,
    ):
        self.static  = StaticPredictor(smoothing=1)
        self.lstm    = LSTMPredictor(smoothing=1)
        self.sw      = static_weight
        self.lw      = lstm_weight
        self.threshold = threshold
        self.smoother  = PredictionSmoother(smoothing)

    def update(self, features: np.ndarray) -> Tuple[str, float]:
        static_label, static_conf = self.static.predict(features)
        self.lstm.push_frame(features)
        lstm_label, lstm_conf     = self.lstm.predict()

        if lstm_label is None:
            label, conf = static_label, static_conf
        elif static_label == lstm_label:
            label = static_label
            conf  = self.sw * static_conf + self.lw * lstm_conf
        else:
            # Weighted vote
            if lstm_conf * self.lw > static_conf * self.sw:
                label, conf = lstm_label, lstm_conf
            else:
                label, conf = static_label, static_conf

        if conf < self.threshold:
            label = UNKNOWN_LABEL

        return self.smoother.update(label, conf)
