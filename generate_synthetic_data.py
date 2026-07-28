"""
generate_synthetic_data.py
==========================
Generates realistic synthetic landmark data for all gesture classes.
Each gesture has a distinct, anatomically-grounded hand pose template
that is then perturbed to simulate natural variation.

Usage:
    python generate_synthetic_data.py --samples 300
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    GESTURE_CLASSES, LANDMARK_FEATURES, CSV_FILE, PROCESSED_DIR
)
from src.dataset import augment_sample

# ─── Landmark index reference (MediaPipe 21-point hand) ──────────────────────
# 0=WRIST  1-4=THUMB  5-8=INDEX  9-12=MIDDLE  13-16=RING  17-20=PINKY

def _make_hand(
    fingers_open: list,          # [thumb, index, middle, ring, pinky] 1=open, 0=closed
    wrist_angle:  float = 0.0,   # global rotation (radians)
    spread:       float = 0.05,  # lateral spread between fingers
) -> np.ndarray:
    """
    Build a 21-landmark hand pose from a high-level finger state vector.
    Returns shape (63,) in normalised coordinates.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # Wrist
    pts[0] = [0.0, 0.0, 0.0]

    # Finger base-to-tip layout (rough anatomical spacing)
    finger_bases_x = [-0.12, -0.06, 0.0, 0.06, 0.12]   # thumb..pinky
    finger_len_y   = [0.18,   0.26,  0.28, 0.24, 0.20]  # open length

    landmark_offset = [1, 5, 9, 13, 17]  # start index per finger

    for fi, (bx, fy) in enumerate(zip(finger_bases_x, finger_len_y)):
        is_open = fingers_open[fi]
        base    = landmark_offset[fi]
        # 4 joints per finger
        for j in range(4):
            t = (j + 1) / 4.0
            if is_open:
                pts[base + j] = [bx + spread * fi * 0.1,
                                  t * fy * 0.9 + 0.08,
                                  -t * 0.02]
            else:
                # Curled: x moves inward, y stays low, z curves forward
                curl = 0.5 + 0.5 * (j / 3.0)
                pts[base + j] = [bx * (1 - curl * 0.6),
                                  0.05 + t * 0.1,
                                  curl * 0.08]

    # Global 2-D rotation
    cos_a, sin_a = np.cos(wrist_angle), np.sin(wrist_angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts[:, :2] = pts[:, :2] @ R.T

    # Normalise
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale

    return pts.flatten()


# ─── Gesture templates ────────────────────────────────────────────────────────

def _wave_hello(n: int) -> np.ndarray:
    """All fingers open, slight variation in wrist angle."""
    base = _make_hand([1, 1, 1, 1, 1], wrist_angle=0.0)
    samples = []
    for _ in range(n):
        angle    = np.random.uniform(-0.4, 0.4)
        s        = _make_hand([1, 1, 1, 1, 1], wrist_angle=angle)
        s       += np.random.normal(0, 0.012, s.shape)
        samples.append(s.astype(np.float32))
    return np.array(samples)


def _hi(n: int) -> np.ndarray:
    """Four fingers open, thumb slightly bent (compact wave)."""
    samples = []
    for _ in range(n):
        angle  = np.random.uniform(-0.2, 0.3)
        thumb  = int(np.random.rand() > 0.5)
        s      = _make_hand([thumb, 1, 1, 1, 1], wrist_angle=angle)
        s     += np.random.normal(0, 0.010, s.shape)
        samples.append(s.astype(np.float32))
    return np.array(samples)


def _stop(n: int) -> np.ndarray:
    """All fingers fully extended, palm facing forward (flat hand)."""
    samples = []
    for _ in range(n):
        angle = np.random.uniform(-0.15, 0.15)
        s     = _make_hand([1, 1, 1, 1, 1],
                            wrist_angle=angle, spread=0.03)
        # Rotate to face camera (tilt back slightly)
        s = s.reshape(-1, 3)
        s[:, 1] *= 1.1
        s[:, 2] += np.random.uniform(-0.05, 0.05)
        s = (s / np.max(np.linalg.norm(s, axis=1))).flatten()
        s += np.random.normal(0, 0.008, s.shape)
        samples.append(s.astype(np.float32))
    return np.array(samples)


def _thank_you(n: int) -> np.ndarray:
    """Flat hand near chin — 4 fingers extended, thumb open, rotated down."""
    samples = []
    for _ in range(n):
        angle = np.random.uniform(0.3, 0.7)   # hand angled toward face
        s     = _make_hand([1, 1, 1, 1, 1], wrist_angle=angle, spread=0.02)
        s    += np.random.normal(0, 0.012, s.shape)
        samples.append(s.astype(np.float32))
    return np.array(samples)


def _yes(n: int) -> np.ndarray:
    """Fist with thumb up."""
    samples = []
    for _ in range(n):
        angle = np.random.uniform(-0.2, 0.2)
        s     = _make_hand([1, 0, 0, 0, 0], wrist_angle=angle)
        s    += np.random.normal(0, 0.010, s.shape)
        samples.append(s.astype(np.float32))
    return np.array(samples)


def _no(n: int) -> np.ndarray:
    """Index finger wagging (only index open, hand tilted)."""
    samples = []
    for _ in range(n):
        angle = np.random.uniform(-0.5, 0.5)
        s     = _make_hand([0, 1, 0, 0, 0], wrist_angle=angle)
        s    += np.random.normal(0, 0.012, s.shape)
        samples.append(s.astype(np.float32))
    return np.array(samples)


GENERATORS = {
    "Hello":    _wave_hello,
    "Hi":       _hi,
    "Stop":     _stop,
    "Thank You": _thank_you,
    "Yes":      _yes,
    "No":       _no,
}


# ─── Main ────────────────────────────────────────────────────────────────────

def generate(samples_per_class: int = 250, augment: bool = True):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cols = [f"f{i}" for i in range(LANDMARK_FEATURES)]

    all_rows = []
    for gesture, gen_fn in GENERATORS.items():
        raw = gen_fn(samples_per_class)
        print(f"  {gesture:12s}: {len(raw)} raw samples", end="")

        if augment:
            aug = []
            for s in raw:
                aug.extend(augment_sample(s))
            raw = np.vstack([raw, aug])
            print(f" + {len(aug)} augmented = {len(raw)}")
        else:
            print()

        df_g = pd.DataFrame(raw, columns=cols)
        df_g.insert(0, "gesture", gesture)
        all_rows.append(df_g)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"\n[Synthetic] Saved {len(df)} total rows → {CSV_FILE}")
    print(f"  Distribution:\n{df['gesture'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic landmark data for gesture recognition"
    )
    parser.add_argument("--samples", type=int, default=250,
                        help="Samples per gesture class (default: 250)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip augmentation step")
    args = parser.parse_args()

    print(f"\n[Synthetic] Generating {args.samples} samples × {len(GESTURE_CLASSES)} gestures")
    generate(args.samples, augment=not args.no_augment)
