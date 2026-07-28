"""
models.py - Deep-learning model architectures for gesture recognition

Two models:
  1. StaticGestureModel  – Dense + BatchNorm + Dropout (per-frame)
  2. LSTMGestureModel    – Bidirectional LSTM for temporal sequences
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Tuple

from src.config import (
    LANDMARK_FEATURES, NUM_CLASSES, NUM_STATIC_CLASSES, SEQUENCE_LENGTH,
    DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, EPOCHS, PATIENCE,
    STATIC_MODEL_PATH, LSTM_MODEL_PATH, LOGS_DIR
)


# ─── Focal Loss (handles class imbalance better than CE) ────────────────────

class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, **kw):
        super().__init__(**kw)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true_oh = tf.one_hot(y_true, NUM_STATIC_CLASSES)
        cross_ent = -y_true_oh * tf.math.log(y_pred)
        pt        = tf.reduce_sum(y_true_oh * y_pred, axis=-1, keepdims=True)
        focal     = self.alpha * tf.pow(1 - pt, self.gamma) * cross_ent
        return tf.reduce_mean(focal)


# ─── Static (per-frame) model ────────────────────────────────────────────────

def build_static_model(
    input_dim: int = LANDMARK_FEATURES,
    num_classes: int = NUM_STATIC_CLASSES,  # 29: A-Z, SPACE, SPEAK, DELETE
) -> keras.Model:
    """
    Dense residual network for single-frame landmark classification.
    Architecture: Input → [Dense-BN-ReLU-Dropout] x3 (with residual skip) → Softmax
    Handles all 29 classes including J, Z, and DELETE.
    """
    inp = keras.Input(shape=(input_dim,), name="landmarks")

    # Block 1
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Block 2 with residual
    h = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(DROPOUT_RATE)(h)
    x = layers.Add()([x, h])          # skip connection

    # Block 3
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.75)(x)

    # Block 4
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5)(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="StaticGestureNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=FocalLoss(name="focal_loss"),
        metrics=["accuracy"],
    )
    return model


# ─── LSTM (sequence) model ────────────────────────────────────────────────────

def build_lstm_model(
    seq_len:    int = SEQUENCE_LENGTH,
    input_dim:  int = LANDMARK_FEATURES,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    Bidirectional LSTM for temporal gesture recognition.
    Architecture: Input → BiLSTM × 2 → Attention → Dense → Softmax
    """
    inp = keras.Input(shape=(seq_len, input_dim), name="landmark_sequence")

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3,
                    recurrent_dropout=0.2)
    )(inp)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3)
    )(x)

    # Self-attention pooling
    attn = layers.Dense(1, activation="tanh")(x)          # (B, T, 1)
    attn = layers.Softmax(axis=1)(attn)                    # (B, T, 1)
    x    = layers.Multiply()([x, attn])                    # (B, T, 128)
    x    = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)  # (B, 128)

    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Dense(64,  activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LSTMGestureNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Training helpers ────────────────────────────────────────────────────────

def get_callbacks(model_path: str, log_dir: str = LOGS_DIR) -> list:
    os.makedirs(log_dir, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, "training_log.csv")
        ),
    ]


def train_static_model(
    X_train, y_train,
    X_val,   y_val,
    class_weights=None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    os.makedirs(LOGS_DIR, exist_ok=True)
    model     = build_static_model()
    callbacks = get_callbacks(STATIC_MODEL_PATH)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(STATIC_MODEL_PATH)
    print(f"[Model] Saved → {STATIC_MODEL_PATH}")
    return model, history


def train_lstm_model(
    X_train, y_train,
    X_val,   y_val,
) -> Tuple[keras.Model, keras.callbacks.History]:
    os.makedirs(LOGS_DIR, exist_ok=True)
    model     = build_lstm_model()
    callbacks = get_callbacks(LSTM_MODEL_PATH)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(LSTM_MODEL_PATH)
    print(f"[Model] LSTM Saved → {LSTM_MODEL_PATH}")
    return model, history


def load_model(path: str) -> keras.Model:
    return keras.models.load_model(path, custom_objects={"FocalLoss": FocalLoss})