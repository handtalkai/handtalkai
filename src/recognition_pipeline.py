"""
recognition_pipeline.py - Modular multi-model prediction registry.

Combines predictions from any number of registered recognizers (alphabet,
number, and future ones such as motion-as-a-registrable-model or common
words) that all consume the same MediaPipe landmark features. Adding a new
model later is a single `.register(...)` call — no changes to the merge
logic itself.
"""

from typing import List, Optional, Tuple

import numpy as np

from src.config import UNKNOWN_LABEL


class MultiModelRecognizer:
    """
    Holds an ordered set of (name, predictor) pairs. Any predictor exposing
    `.predict_with_top_k(features, k) -> ((label, confidence), top_k_list)`
    can be registered — duck-typed, not tied to a concrete predictor class.

    Each frame, every registered predictor runs against the same feature
    vector (the single MediaPipe extraction already performed upstream — no
    landmark processing is duplicated here). Whichever predictor returns the
    highest-confidence non-"Unknown" label wins.
    """

    def __init__(self):
        self._models: List[Tuple[str, object]] = []

    def register(self, name: str, predictor) -> None:
        self._models.append((name, predictor))

    def predict_with_top_k(
        self, features: np.ndarray, k: int = 3
    ) -> Tuple[str, float, list, Optional[str]]:
        """Returns (label, confidence, top_k, source_model_name)."""
        candidates = []
        for name, predictor in self._models:
            (label, conf), top_k = predictor.predict_with_top_k(features, k=k)
            if label != UNKNOWN_LABEL:
                candidates.append((name, label, conf, top_k))

        if not candidates:
            return UNKNOWN_LABEL, 0.0, [], None

        name, label, conf, top_k = max(candidates, key=lambda c: c[2])
        return label, conf, top_k, name
