"""
number_model.py - Dense model for ASL digit (0-9) static gesture recognition

Self-contained: does not import src/dataset.py or src/models.py's DatasetLoader/
FocalLoss (both are hardcoded to the alphabet's class list), so training or
loading the number model can never touch the alphabet model's files or code
path. Mirrors the structure of src/motion_model.py, which follows the same
"self-contained per-model-type module" pattern for J/Z.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    NUMBER_CSV_FILE, NUMBER_MODEL_PATH, NUMBER_ENCODER_PATH,
    NUMBER_CLASSES, NUM_NUMBER_CLASSES, LANDMARK_FEATURES,
    DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, EPOCHS, PATIENCE,
    TEST_SPLIT, VAL_SPLIT, RANDOM_STATE, MODELS_DIR, LOGS_DIR,
)
from src.models import get_callbacks


def build_number_model(
    input_dim: int = LANDMARK_FEATURES,
    num_classes: int = NUM_NUMBER_CLASSES,
) -> keras.Model:
    """Dense residual network for single-frame digit landmark classification."""
    inp = keras.Input(shape=(input_dim,), name="landmarks")

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    h = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(DROPOUT_RATE)(h)
    x = layers.Add()([x, h])          # skip connection

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5)(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="NumberGestureNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _load_number_dataset(csv_path: str = NUMBER_CSV_FILE):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    le.classes_ = np.array(NUMBER_CLASSES)

    X = df.drop(columns=["gesture"]).values.astype(np.float32)
    # Digit labels ("0".."9") read back from CSV can be inferred as ints by
    # pandas; force back to str so they match NUMBER_CLASSES' string classes.
    y = le.transform(df["gesture"].astype(str).values)
    print(f"[Number] Loaded {len(X)} samples | classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=RANDOM_STATE, stratify=y_train,
    )
    print(f"[Number] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def train_number_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, le = _load_number_dataset()

    model = build_number_model()
    model.summary()

    # Separate log subdir so this never overwrites the alphabet model's
    # shared logs/training_log.csv.
    number_log_dir = os.path.join(LOGS_DIR, "number")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(NUMBER_MODEL_PATH, log_dir=number_log_dir),
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = float(np.mean(y_pred == y_test))
    print(f"\n[Number] Test accuracy: {acc:.4f}")
    for i, digit in enumerate(NUMBER_CLASSES):
        mask = y_test == i
        if mask.any():
            print(f"  {digit}: {np.mean(y_pred[mask] == y_test[mask]):.4f}")

    model.save(NUMBER_MODEL_PATH)
    np.save(NUMBER_ENCODER_PATH, le.classes_)
    print(f"[Number] Saved -> {NUMBER_MODEL_PATH}")
    return model, history, acc
