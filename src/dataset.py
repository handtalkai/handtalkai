"""
dataset.py - Gesture dataset recording, augmentation, and loading
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import rotate as scipy_rotate

from src.config import (
    GESTURE_CLASSES, STATIC_CLASSES, CSV_FILE, SEQUENCE_FILE, LABEL_ENCODER_PATH,
    TEST_SPLIT, VAL_SPLIT, RANDOM_STATE,
    AUGMENTATION_FACTOR, NOISE_SCALE, ROTATION_RANGE, SCALE_RANGE,
    SEQUENCE_LENGTH, LANDMARK_FEATURES, PROCESSED_DIR, MODELS_DIR
)
from src.landmark_extractor import HandLandmarkExtractor


# ─── Augmentation helpers ────────────────────────────────────────────────────

def _jitter(sample: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, NOISE_SCALE, sample.shape).astype(np.float32)
    return sample + noise


def _rotate_2d(sample: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate x,y landmark coordinates around the wrist (index 0)."""
    pts = sample.reshape(-1, 3).copy()
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts[:, :2] = pts[:, :2] @ R.T
    return pts.flatten()


def _scale(sample: np.ndarray, factor: float) -> np.ndarray:
    pts = sample.reshape(-1, 3).copy()
    pts *= factor
    return pts.flatten()


def augment_sample(sample: np.ndarray) -> List[np.ndarray]:
    """Return a list of augmented variants (not including original)."""
    variants = []
    for _ in range(AUGMENTATION_FACTOR - 1):
        s = sample.copy()
        s = _jitter(s)
        angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        s = _rotate_2d(s, angle)
        scale = np.random.uniform(*SCALE_RANGE)
        s = _scale(s, scale)
        # Re-normalize
        norm = np.linalg.norm(s.reshape(-1, 3), axis=1).max()
        if norm > 0:
            s = (s.reshape(-1, 3) / norm).flatten()
        variants.append(s)
    return variants


# ─── DatasetRecorder ────────────────────────────────────────────────────────

class DatasetRecorder:
    """Records hand landmarks from webcam and saves them to CSV."""

    def __init__(self, gesture: str, target_samples: int = 200):
        self.gesture        = gesture
        self.target_samples = target_samples
        self.extractor      = HandLandmarkExtractor(max_hands=1)
        self.samples: List[np.ndarray] = []

    def record(self, cap: cv2.VideoCapture) -> int:
        """
        Block until target_samples are collected.
        Returns number of samples recorded.
        Press 'q' to stop early.
        """
        print(f"\n[Recorder] Collecting '{self.gesture}' — target: {self.target_samples}")
        print("  Press SPACE to start / pause, Q to quit.")

        collecting = False
        while len(self.samples) < self.target_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            features, results = self.extractor.extract_features(frame)
            self.extractor.draw_landmarks(frame, results)

            # HUD
            pct = len(self.samples) / self.target_samples
            bar_w = int(400 * pct)
            cv2.rectangle(frame, (10, 10), (410, 35), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, 10), (10 + bar_w, 35), (0, 220, 100), -1)
            cv2.putText(frame, f"Gesture: {self.gesture}  [{len(self.samples)}/{self.target_samples}]",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status = "COLLECTING" if collecting else "PAUSED — press SPACE"
            cv2.putText(frame, status, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 100) if collecting else (0, 120, 255), 2)

            if collecting and features is not None:
                self.samples.append(features)

            cv2.imshow("Data Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                collecting = not collecting
            elif key == ord('q'):
                break

        cv2.destroyWindow("Data Recorder")
        self.extractor.close()
        return len(self.samples)

    def save_to_csv(self, csv_path: str = CSV_FILE):
        """Append samples (+ augmented) to the master CSV."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        all_samples = []
        for s in self.samples:
            all_samples.append(s)
            all_samples.extend(augment_sample(s))

        cols = [f"f{i}" for i in range(LANDMARK_FEATURES)]
        df = pd.DataFrame(all_samples, columns=cols)
        df.insert(0, "gesture", self.gesture)

        mode   = 'a' if os.path.exists(csv_path) else 'w'
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode=mode, header=header, index=False)
        print(f"[Dataset] Saved {len(all_samples)} rows for '{self.gesture}' → {csv_path}")


# ─── DatasetLoader ───────────────────────────────────────────────────────────

class DatasetLoader:
    """Loads, encodes, and splits the landmark CSV dataset."""

    def __init__(self, csv_path: str = CSV_FILE):
        self.csv_path = csv_path
        self.le       = LabelEncoder()
        self.le.classes_ = np.array(STATIC_CLASSES)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.csv_path)
        X  = df.drop(columns=["gesture"]).values.astype(np.float32)
        y  = self.le.transform(df["gesture"].values)
        print(f"[Dataset] Loaded {len(X)} samples | classes: {list(self.le.classes_)}")
        return X, y

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=VAL_SPLIT / (1 - TEST_SPLIT),
            random_state=RANDOM_STATE, stratify=y_train
        )
        print(f"[Dataset] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_label_encoder(self, path: str = LABEL_ENCODER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.le.classes_)

    @staticmethod
    def load_label_encoder(path: str = LABEL_ENCODER_PATH) -> LabelEncoder:
        le = LabelEncoder()
        le.classes_ = np.load(path, allow_pickle=True)
        return le

    @property
    def class_names(self) -> List[str]:
        return list(self.le.classes_)

    # ─── Class balance report ────────────────────────────────────────────────
    def class_distribution(self) -> pd.Series:
        df = pd.read_csv(self.csv_path)
        return df["gesture"].value_counts()


# ─── Sequence dataset (for LSTM) ─────────────────────────────────────────────

def build_sequence_dataset(
    csv_path:  str = CSV_FILE,
    seq_len:   int = SEQUENCE_LENGTH,
    save_path: str = SEQUENCE_FILE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert per-frame landmark CSV into overlapping windows of length seq_len.
    Saves and returns (X_seq, y_seq).
    """
    df = pd.read_csv(csv_path)
    le = LabelEncoder().fit(STATIC_CLASSES)

    X_seqs, y_seqs = [], []
    for gesture, group in df.groupby("gesture"):
        feats = group.drop(columns=["gesture"]).values.astype(np.float32)
        label = le.transform([gesture])[0]
        # Sliding window
        for start in range(0, len(feats) - seq_len + 1, seq_len // 2):
            window = feats[start: start + seq_len]
            if len(window) == seq_len:
                X_seqs.append(window)
                y_seqs.append(label)

    X_seq = np.array(X_seqs, dtype=np.float32)
    y_seq = np.array(y_seqs, dtype=np.int32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X_seq, y=y_seq, classes=le.classes_)
    print(f"[Sequence] {len(X_seq)} sequences ({seq_len} frames) → {save_path}")
    return X_seq, y_seq