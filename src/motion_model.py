"""
motion_model.py - BiLSTM model for J & Z motion recognition
"""

import collections
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from src.config import (
    MOTION_CSV_FILE, MOTION_MODEL_PATH, MOTION_ENCODER_PATH,
    MOTION_CLASSES, MOTION_SEQUENCE_LENGTH, LANDMARK_FEATURES,
    NUM_MOTION_CLASSES, EPOCHS, DROPOUT_RATE,
    LEARNING_RATE, PATIENCE, TEST_SPLIT, VAL_SPLIT, RANDOM_STATE,
    MODELS_DIR, LOGS_DIR,
)


def build_motion_model(seq_len=MOTION_SEQUENCE_LENGTH,
                       feat_dim=LANDMARK_FEATURES,
                       num_classes=NUM_MOTION_CLASSES):
    inp = keras.Input(shape=(seq_len, feat_dim), name="motion_seq")

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3,
                    recurrent_dropout=0.2))(inp)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.3))(x)

    # Self-attention — use GlobalAveragePooling instead of Lambda
    attn   = layers.Dense(1, activation="tanh")(x)        # (B, T, 1)
    attn   = layers.Softmax(axis=1)(attn)                  # (B, T, 1)
    x      = layers.Multiply()([x, attn])                  # (B, T, 64)
    x      = layers.GlobalAveragePooling1D()(x)            # (B, 64)

    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(num_classes, activation="softmax", name="out")(x)

    model = keras.Model(inputs=inp, outputs=out, name="MotionNet")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_sequences():
    df        = pd.read_csv(MOTION_CSV_FILE)
    le        = LabelEncoder()
    le.classes_ = np.array(MOTION_CLASSES)
    feat_cols = [f"f{i}" for i in range(LANDMARK_FEATURES)]
    X_seqs, y_seqs = [], []

    for (gesture, clip_id), group in df.groupby(["gesture", "clip_id"]):
        group  = group.sort_values("frame_idx")
        frames = group[feat_cols].values.astype(np.float32)
        # Pad or trim to exact sequence length
        if len(frames) < MOTION_SEQUENCE_LENGTH:
            pad    = np.zeros((MOTION_SEQUENCE_LENGTH - len(frames),
                               LANDMARK_FEATURES), np.float32)
            frames = np.vstack([frames, pad])
        frames = frames[:MOTION_SEQUENCE_LENGTH]
        X_seqs.append(frames)
        y_seqs.append(le.transform([gesture])[0])

    X = np.array(X_seqs, np.float32)
    y = np.array(y_seqs, np.int32)

    # Augment x3 with small noise
    Xa, ya = list(X), list(y)
    for xi, yi in zip(X, y):
        for _ in range(2):
            Xa.append(xi + np.random.normal(0, 0.008, xi.shape).astype(np.float32))
            ya.append(yi)
    X = np.array(Xa, np.float32)
    y = np.array(ya, np.int32)

    print(f"[Motion] {len(X)} sequences | "
          f"J={np.sum(y==0)}  Z={np.sum(y==1)}")
    return X, y, le


def train_motion_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    X, y, le = load_sequences()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SPLIT,
        random_state=RANDOM_STATE, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=RANDOM_STATE, stratify=y_tr)

    print(f"[Motion] Train={len(X_tr)}  Val={len(X_val)}  Test={len(X_te)}")

    model = build_motion_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            MOTION_MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, verbose=1),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=16,
        callbacks=callbacks, verbose=1,
    )

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    acc    = float(np.mean(y_pred == y_te))
    print(f"\n[Motion] Test accuracy: {acc:.4f}")
    for i, ltr in enumerate(MOTION_CLASSES):
        mask = y_te == i
        if mask.any():
            print(f"  {ltr}: {np.mean(y_pred[mask]==y_te[mask]):.4f}")

    np.save(MOTION_ENCODER_PATH, le.classes_)
    model.save(MOTION_MODEL_PATH)
    print(f"[Motion] Saved -> {MOTION_MODEL_PATH}")
    return model, history, acc


class MotionPredictor:
    """
    Rolls a 60-frame buffer and predicts J or Z when motion is detected.
    Only fires when wrist movement exceeds threshold — avoids false
    positives when hand is held in a static pose.
    """

    def __init__(self, model_path=MOTION_MODEL_PATH,
                 encoder_path=MOTION_ENCODER_PATH,
                 seq_len=MOTION_SEQUENCE_LENGTH,
                 threshold=0.85):
        self.model     = keras.models.load_model(model_path)
        le             = LabelEncoder()
        le.classes_    = np.load(encoder_path, allow_pickle=True)
        self.le        = le
        self.seq_len   = seq_len
        self.threshold = threshold
        self._buf          = collections.deque(maxlen=seq_len)
        self._prev_wrist   = None
        self._wrist_diffs  = collections.deque(maxlen=seq_len - 1)
        self._motion_total = 0.0

    def push(self, features: np.ndarray):
        new_wrist = features[:3].copy()
        if self._prev_wrist is not None:
            diff_norm = float(np.linalg.norm(new_wrist - self._prev_wrist))
            if len(self._wrist_diffs) == self.seq_len - 1:
                self._motion_total -= self._wrist_diffs[0]
            self._wrist_diffs.append(diff_norm)
            self._motion_total += diff_norm
        self._buf.append(features)
        self._prev_wrist = new_wrist

    def _has_motion(self) -> bool:
        return len(self._buf) >= self.seq_len and self._motion_total > 0.05

    def predict(self):
        if not self._has_motion():
            return None, 0.0
        seq   = np.array(self._buf, np.float32)[np.newaxis]
        probs = self.model(seq, training=False).numpy()[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        if conf < self.threshold:
            return None, 0.0
        return self.le.classes_[idx], conf

    def reset(self):
        self._buf          = collections.deque(maxlen=self.seq_len)
        self._prev_wrist   = None
        self._wrist_diffs  = collections.deque(maxlen=self.seq_len - 1)
        self._motion_total = 0.0