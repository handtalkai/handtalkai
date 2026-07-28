"""
landmark_extractor.py - MediaPipe hand landmark extraction and normalization
"""

import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
from src.config import (
    NUM_LANDMARKS, LANDMARK_FEATURES,
    MP_DETECTION_CONF, MP_TRACKING_CONF
)


class HandLandmarkExtractor:
    """
    Wraps MediaPipe Hands and provides normalized landmark extraction.
    Handles single- and multi-hand detection, coordinate normalization,
    and feature vector construction.
    """

    def __init__(self, max_hands: int = 2, static_mode: bool = False):
        self.mp_hands   = mp.solutions.hands
        self.mp_draw    = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode        = static_mode,
            max_num_hands            = max_hands,
            min_detection_confidence = MP_DETECTION_CONF,
            min_tracking_confidence  = MP_TRACKING_CONF,
        )

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _normalize_landmarks(self, landmarks) -> np.ndarray:
        """
        Translate so wrist (landmark 0) is the origin, then scale so the
        maximum Euclidean distance from the wrist equals 1.
        Returns a flat float32 array of shape (63,).
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                       dtype=np.float32)

        # Translate
        origin = pts[0].copy()
        pts -= origin

        # Scale
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts /= scale

        return pts.flatten()

    # ─── Public API ─────────────────────────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray):
        """
        Run MediaPipe on a BGR frame.
        Returns the raw MediaPipe results object.
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return results

    def extract_features(
        self,
        bgr_frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], any]:
        """
        Returns (feature_vector | None, mediapipe_results).
        feature_vector shape: (63,)  – first detected hand only.
        """
        results = self.process_frame(bgr_frame)
        if not results.multi_hand_landmarks:
            return None, results

        # Use the first (most confident) hand
        hand_landmarks = results.multi_hand_landmarks[0]
        features = self._normalize_landmarks(hand_landmarks)
        return features, results

    def extract_all_hands(
        self,
        bgr_frame: np.ndarray,
    ) -> Tuple[List[np.ndarray], any]:
        """
        Returns (list_of_feature_vectors, mediapipe_results).
        Each vector shape: (63,).
        """
        results = self.process_frame(bgr_frame)
        if not results.multi_hand_landmarks:
            return [], results

        vectors = [
            self._normalize_landmarks(hl)
            for hl in results.multi_hand_landmarks
        ]
        return vectors, results

    def draw_landmarks(
        self,
        frame: np.ndarray,
        results,
        color_override: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Draw hand skeleton on frame in-place. Returns the frame.
        """
        if not results.multi_hand_landmarks:
            return frame

        for hand_landmarks in results.multi_hand_landmarks:
            if color_override:
                # Custom colour drawing
                import cv2
                connections = self.mp_hands.HAND_CONNECTIONS
                h, w = frame.shape[:2]
                for connection in connections:
                    start_idx, end_idx = connection
                    start = hand_landmarks.landmark[start_idx]
                    end   = hand_landmarks.landmark[end_idx]
                    sx, sy = int(start.x * w), int(start.y * h)
                    ex, ey = int(end.x   * w), int(end.y   * h)
                    cv2.line(frame, (sx, sy), (ex, ey), color_override, 2)
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, color_override, -1)
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1)
            else:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

        return frame

    def close(self):
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
