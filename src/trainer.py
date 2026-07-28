"""
trainer.py - End-to-end training pipeline with evaluation and reporting
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    GESTURE_CLASSES, STATIC_CLASSES, REPORTS_DIR, LOGS_DIR,
    CSV_FILE, SEQUENCE_FILE, LABEL_ENCODER_PATH,
    STATIC_MODEL_PATH, LSTM_MODEL_PATH
)
from src.dataset import DatasetLoader, build_sequence_dataset
from src.models import (
    train_static_model, train_lstm_model, load_model
)


class GestureTrainer:
    """Orchestrates data loading, training, evaluation, and report generation."""

    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self.loader = DatasetLoader(CSV_FILE)

    # ─── Static model ────────────────────────────────────────────────────────

    def train_static(self) -> dict:
        print("\n" + "=" * 60)
        print(" TRAINING — Static Gesture Model")
        print("=" * 60)

        X, y = self.loader.load()
        X_train, X_val, X_test, y_train, y_val, y_test = self.loader.split(X, y)

        # Class weights for imbalanced datasets
        cw = compute_class_weight("balanced",
                                  classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(cw))

        model, history = train_static_model(X_train, y_train, X_val, y_val,
                                            class_weights=class_weights)
        self.loader.save_label_encoder(LABEL_ENCODER_PATH)

        results = self._evaluate(model, X_test, y_test, tag="static")
        self._plot_history(history, tag="static")
        self._save_report(results, history, tag="static")
        return results

    # ─── LSTM model ──────────────────────────────────────────────────────────

    def train_lstm(self) -> dict:
        print("\n" + "=" * 60)
        print(" TRAINING — LSTM Sequence Model")
        print("=" * 60)

        X_seq, y_seq = build_sequence_dataset()
        from sklearn.model_selection import train_test_split
        from src.config import TEST_SPLIT, VAL_SPLIT, RANDOM_STATE

        X_tr, X_test, y_tr, y_test = train_test_split(
            X_seq, y_seq, test_size=TEST_SPLIT,
            random_state=RANDOM_STATE, stratify=y_seq
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr,
            test_size=VAL_SPLIT / (1 - TEST_SPLIT),
            random_state=RANDOM_STATE, stratify=y_tr
        )

        model, history = train_lstm_model(X_tr, y_tr, X_val, y_val)

        results = self._evaluate(model, X_test, y_test, tag="lstm")
        self._plot_history(history, tag="lstm")
        self._save_report(results, history, tag="lstm")
        return results

    # ─── Evaluation ──────────────────────────────────────────────────────────

    def _evaluate(self, model, X_test, y_test, tag: str) -> dict:
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred      = np.argmax(y_pred_prob, axis=1)
        acc         = accuracy_score(y_test, y_pred)

        print(f"\n[{tag.upper()}] Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=STATIC_CLASSES))

        self._plot_confusion_matrix(y_test, y_pred, tag=tag)

        return {
            "accuracy": float(acc),
            "report":   classification_report(
                            y_test, y_pred,
                            target_names=STATIC_CLASSES,
                            output_dict=True
                        ),
        }

    # ─── Plots ───────────────────────────────────────────────────────────────

    def _plot_history(self, history, tag: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Training History — {tag.upper()} Model",
                     fontsize=14, fontweight="bold")

        # Accuracy
        axes[0].plot(history.history["accuracy"],     label="Train", linewidth=2)
        axes[0].plot(history.history["val_accuracy"], label="Val",   linewidth=2)
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history["loss"],     label="Train", linewidth=2)
        axes[1].plot(history.history["val_loss"], label="Val",   linewidth=2)
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, f"history_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Report] Saved history plot → {path}")

    def _plot_confusion_matrix(self, y_true, y_pred, tag: str):
        cm   = confusion_matrix(y_true, y_pred)
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Confusion Matrix — {tag.upper()} Model",
                     fontsize=14, fontweight="bold")

        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=STATIC_CLASSES, yticklabels=STATIC_CLASSES,
                    ax=axes[0])
        axes[0].set_title("Raw Counts")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        # Normalised
        sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=STATIC_CLASSES, yticklabels=STATIC_CLASSES,
                    ax=axes[1])
        axes[1].set_title("Normalised")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, f"confusion_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Report] Saved confusion matrix → {path}")

    # ─── JSON report ─────────────────────────────────────────────────────────

    def _save_report(self, results: dict, history, tag: str):
        report = {
            "tag":          tag,
            "test_accuracy": results["accuracy"],
            "epochs_run":    len(history.history["accuracy"]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
            "classification_report": results["report"],
        }
        path = os.path.join(REPORTS_DIR, f"report_{tag}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Report] Saved JSON report → {path}")

    # ─── Dataset stats ───────────────────────────────────────────────────────

    def print_dataset_stats(self):
        dist = self.loader.class_distribution()
        print("\n[Dataset] Class distribution:")
        print(dist.to_string())
        self._plot_class_dist(dist)

    def _plot_class_dist(self, dist: pd.Series):
        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = plt.cm.Set2(np.linspace(0, 1, len(dist)))
        bars    = ax.bar(dist.index, dist.values, color=colors, edgecolor="black")
        ax.bar_label(bars, padding=3)
        ax.set_title("Dataset Class Distribution")
        ax.set_xlabel("Gesture")
        ax.set_ylabel("Samples")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "class_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Report] Saved class distribution → {path}")