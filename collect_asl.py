"""
collect_asl.py - Guided ASL A-Z + SPACE fingerspelling data collection
=======================================================================
Walks you through all 26 letters + SPACE one by one.
Shows an on-screen reference diagram + live webcam + progress bar.

Usage:
    py collect_asl.py                       # collect all 26 letters + SPACE
    py collect_asl.py --start G             # resume from letter G
    py collect_asl.py --letter SPACE        # collect only SPACE
    py collect_asl.py --samples 300         # 300 samples per letter
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    GESTURE_CLASSES, MOTION_LETTERS, LANDMARK_FEATURES,
    CSV_FILE, PROCESSED_DIR
)
from src.landmark_extractor import HandLandmarkExtractor
from src.dataset import augment_sample

# A-Z only — this rebuild excludes SPACE/SPEAK/DELETE
ALPHA_LETTERS = [c for c in GESTURE_CLASSES if len(c) == 1]

# Quality thresholds
BLUR_THRESHOLD       = 80.0   # Laplacian variance below this = too blurry
DUPLICATE_THRESHOLD  = 0.01   # min L2 distance between consecutive saved samples
COUNTDOWN_SECONDS    = 3

# ─── ASL fingerspelling descriptions & finger states ─────────────────────────
# Each entry: (description, tip_color_hint)
ASL_HINTS = {
    "A": "Fist with thumb resting beside fingers (not tucked)",
    "B": "Four fingers straight up, thumb folded across palm",
    "C": "All fingers curved into a C shape",
    "D": "Index up, other fingers touch thumb forming a circle",
    "DELETE": "Closed fist — all four fingers folded over thumb, like letter S but thumb tucked under",
    "E": "All fingers bent/hooked down, thumb tucked under",
    "F": "Index + thumb touch (OK), other 3 fingers up",
    "G": "Index points sideways, thumb parallel (like a gun sideways)",
    "H": "Index + middle point sideways together, horizontal",
    "I": "Pinky up, fist closed, other fingers down",
    "J": "Pinky up (letter I pose) — hold still, no movement needed",
    "K": "Index up, middle up at angle, thumb between them",
    "L": "L-shape: index points up, thumb points out sideways",
    "M": "Thumb tucked under first 3 fingers (index, middle, ring)",
    "N": "Thumb tucked under first 2 fingers (index, middle)",
    "O": "All fingers curve to meet thumb forming an O",
    "P": "Like K but pointing downward",
    "Q": "Like G but pointing downward",
    "R": "Index + middle crossed (fingers crossed for luck)",
    "S": "Fist with thumb over fingers",
    "T": "Thumb tucked between index and middle finger",
    "U": "Index + middle up together, parallel",
    "V": "Index + middle up in a V / peace sign",
    "W": "Index + middle + ring up, spread apart",
    "X": "Index finger hooked / bent like a hook",
    "Y": "Thumb + pinky out, other fingers folded",
    "Z": "Index finger pointing up/out — hold still, no movement needed",
    "SPACE": "Flat open hand (B shape), palm facing sideways — signals a word break",
    "SPEAK": "THUMBS UP — thumb pointing straight up, all 4 fingers fully folded into fist",
}

# Confusable letter groups — shown as warning
CONFUSABLE = {
    "A": ["S", "E"],
    "B": ["4"],
    "D": ["G", "F"],
    "DELETE": ["A", "S", "E"],
    "E": ["A", "S"],
    "F": ["D", "9"],
    "G": ["D", "Q"],
    "H": ["U", "N"],
    "J": ["I"],
    "K": ["P", "V"],
    "M": ["N", "S"],
    "N": ["M", "H"],
    "P": ["K", "Q"],
    "Q": ["G", "P"],
    "R": ["U", "V"],
    "S": ["A", "E"],
    "T": ["N", "D"],
    "U": ["H", "R"],
    "V": ["U", "K"],
    "Z": ["G", "D"],
}

# ─── Drawing helpers ──────────────────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_S     = cv2.FONT_HERSHEY_SIMPLEX
GREEN      = (0, 220, 100)
YELLOW     = (0, 220, 220)
RED        = (60,  60, 220)
WHITE      = (240, 240, 240)
DARK       = (20,  20,  25)
PANEL_BG   = (30,  30,  40)


def draw_letter_panel(frame, letter: str, collected: int, target: int,
                      total_done: int, total: int, recording: bool):
    h, w = frame.shape[:2]

    # ── Right panel background ───────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 320, 0), (w, h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (w - 320, 0), (w - 320, h), GREEN, 2)

    px = w - 305

    # ── Big letter display ───────────────────────────────────────────────────
    cv2.putText(frame, "LETTER", (px, 40), FONT_S, 0.6, (120, 120, 130), 1, cv2.LINE_AA)
    cv2.putText(frame, letter,   (px, 115), FONT, 3.2, GREEN, 5, cv2.LINE_AA)

    # ── Overall progress ─────────────────────────────────────────────────────
    cv2.putText(frame, f"Overall: {total_done}/26 letters",
                (px, 145), FONT_S, 0.55, (150, 150, 160), 1, cv2.LINE_AA)

    # ── Hint ─────────────────────────────────────────────────────────────────
    cv2.putText(frame, "HOW TO SIGN:", (px, 175), FONT_S, 0.55, YELLOW, 1, cv2.LINE_AA)
    hint = ASL_HINTS.get(letter, "")
    # Word-wrap hint to ~35 chars
    words, line, lines = hint.split(), "", []
    for w_txt in words:
        if len(line) + len(w_txt) + 1 <= 34:
            line += (" " if line else "") + w_txt
        else:
            lines.append(line)
            line = w_txt
    if line:
        lines.append(line)
    for i, l in enumerate(lines[:3]):
        cv2.putText(frame, l, (px, 198 + i * 22),
                    FONT_S, 0.52, WHITE, 1, cv2.LINE_AA)

    # ── Confusable warning ────────────────────────────────────────────────────
    if letter in CONFUSABLE:
        similar = ", ".join(CONFUSABLE[letter])
        cv2.putText(frame, f"Watch out: similar to {similar}",
                    (px, 275), FONT_S, 0.48, (80, 80, 220), 1, cv2.LINE_AA)

    # ── Sample progress bar ──────────────────────────────────────────────────
    cv2.putText(frame, "SAMPLES", (px, 310), FONT_S, 0.55, (120, 120, 130), 1, cv2.LINE_AA)
    bar_w = 280
    pct   = min(collected / target, 1.0)
    cv2.rectangle(frame, (px, 318), (px + bar_w, 340), (50, 50, 60), -1)
    fill_color = GREEN if pct < 0.99 else (0, 180, 255)
    cv2.rectangle(frame, (px, 318), (px + int(bar_w * pct), 340), fill_color, -1)
    cv2.putText(frame, f"{collected} / {target}",
                (px, 360), FONT_S, 0.65, WHITE, 1, cv2.LINE_AA)

    # ── Recording indicator ──────────────────────────────────────────────────
    if recording:
        cv2.circle(frame, (px + 10, 385), 10, RED, -1)
        cv2.putText(frame, "RECORDING", (px + 28, 391),
                    FONT_S, 0.65, RED, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "PAUSED  — press SPACE",
                    (px, 391), FONT_S, 0.55, YELLOW, 1, cv2.LINE_AA)

    # ── Controls ─────────────────────────────────────────────────────────────
    y = h - 130
    for key, desc in [("SPACE", "Start / Pause"), ("N", "Skip letter"),
                       ("B", "Redo letter"), ("Q", "Save & Quit")]:
        cv2.putText(frame, f"[{key}]", (px, y), FONT_S, 0.5, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, desc,        (px + 70, y), FONT_S, 0.5, (160, 160, 170), 1, cv2.LINE_AA)
        y += 24

    return frame


def draw_hand_missing(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "NO HAND DETECTED",
                (20, h // 2), FONT_S, 1.0, (60, 60, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, "Show your hand to the camera",
                (20, h // 2 + 35), FONT_S, 0.65, YELLOW, 1, cv2.LINE_AA)


def draw_reject_reason(frame, reason: str):
    h, w = frame.shape[:2]
    cv2.putText(frame, reason, (20, h // 2 + 70), FONT_S, 0.7, RED, 2, cv2.LINE_AA)


def draw_countdown(frame, seconds_left: float):
    h, w = frame.shape[:2]
    n = int(np.ceil(seconds_left))
    if n <= 0:
        return
    text = str(n)
    (tw, th), _ = cv2.getTextSize(text, FONT, 4.0, 6)
    cx, cy = (w - 320) // 2, h // 2
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                FONT, 4.0, YELLOW, 6, cv2.LINE_AA)
    cv2.putText(frame, "GET READY", (cx - 90, cy - th),
                FONT_S, 0.9, WHITE, 2, cv2.LINE_AA)


def is_blurry(frame) -> bool:
    """True if the frame's Laplacian variance is below BLUR_THRESHOLD."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def is_duplicate(features: np.ndarray, last_features: Optional[np.ndarray]) -> bool:
    """True if features are nearly identical to the last saved sample."""
    if last_features is None:
        return False
    return float(np.linalg.norm(features - last_features)) < DUPLICATE_THRESHOLD


# ─── Collection logic ─────────────────────────────────────────────────────────

def collect_letter(
    cap,
    extractor: HandLandmarkExtractor,
    letter: str,
    target: int,
    letter_index: int,
    total_letters: int,
) -> list:
    """
    Collect `target` landmark samples for one letter.
    Returns list of np.ndarray feature vectors.
    """
    samples       = []
    last_features = None
    recording     = False
    countdown_end = None   # timestamp when countdown finishes, or None if not counting down

    while len(samples) < target:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        blurry    = is_blurry(frame)
        results   = extractor.process_frame(frame)
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        features = None
        reject_reason = None
        if num_hands == 0:
            draw_hand_missing(frame)
        elif num_hands > 1:
            reject_reason = "MULTIPLE HANDS — show only one hand"
        else:
            features = extractor._normalize_landmarks(results.multi_hand_landmarks[0])

        extractor.draw_landmarks(frame, results, color_override=(0, 200, 100))

        # ── Countdown gate before recording actually starts ──────────────────
        in_countdown = countdown_end is not None
        if in_countdown:
            remaining = countdown_end - time.time()
            if remaining <= 0:
                countdown_end = None
                recording = True
            else:
                draw_countdown(frame, remaining)

        if reject_reason:
            draw_reject_reason(frame, reject_reason)
            recording = False

        draw_letter_panel(frame, letter, len(samples), target,
                          letter_index, total_letters, recording and not in_countdown)

        if recording and not in_countdown and features is not None and not reject_reason:
            if blurry:
                draw_reject_reason(frame, "TOO BLURRY — hold steady")
            elif is_duplicate(features, last_features):
                pass  # near-identical to last sample, skip silently
            else:
                samples.append(features)
                last_features = features
                if len(samples) % 10 == 0:
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (frame.shape[1] - 320, frame.shape[0]),
                                  (0, 80, 0), -1)
                    cv2.addWeighted(flash, 0.15, frame, 0.85, 0, frame)

        cv2.imshow("ASL Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if recording or in_countdown:
                recording = False
                countdown_end = None
            else:
                countdown_end = time.time() + COUNTDOWN_SECONDS
        elif key == ord('n'):
            print(f"  [Skip] {letter}")
            return []          # skip = return empty
        elif key == ord('b'):
            print(f"  [Redo] {letter}")
            samples = []
            last_features = None
            recording = False
            countdown_end = None
        elif key == ord('q'):
            return None        # None = quit everything

    return samples


def save_samples(samples: list, letter: str):
    """Save + augment samples to CSV."""
    if not samples:
        return 0

    all_rows = []
    for s in samples:
        all_rows.append(s)
        all_rows.extend(augment_sample(s))

    cols = [f"f{i}" for i in range(LANDMARK_FEATURES)]
    df   = pd.DataFrame(all_rows, columns=cols)
    df.insert(0, "gesture", letter)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    mode   = 'a' if os.path.exists(CSV_FILE) else 'w'
    header = not os.path.exists(CSV_FILE)
    df.to_csv(CSV_FILE, mode=mode, header=header, index=False)
    return len(all_rows)


def already_collected(letter: str) -> bool:
    """Check if this letter already exists in the CSV."""
    if not os.path.exists(CSV_FILE):
        return False
    df = pd.read_csv(CSV_FILE)
    return letter in df["gesture"].values


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Guided ASL A-Z data collection")
    parser.add_argument("--start",   default="A",
                        help="Letter to start from (e.g. --start G)")
    parser.add_argument("--letter",  default=None,
                        help="Collect only one specific letter")
    parser.add_argument("--samples", type=int, default=400,
                        help="Samples per letter (default: 400, recommended range 300-500)")
    parser.add_argument("--redo",    action="store_true",
                        help="Re-collect letters even if already in dataset")
    args = parser.parse_args()

    # Determine which letters to collect (A-Z only for this rebuild)
    if args.letter:
        letters = [args.letter.upper()]
    else:
        start_idx = ALPHA_LETTERS.index(args.start.upper()) \
                    if args.start.upper() in ALPHA_LETTERS else 0
        letters   = ALPHA_LETTERS[start_idx:]

    print("\n" + "=" * 55)
    print(f"  ASL Fingerspelling Data Collection")
    print(f"  Letters: {' '.join(letters)}")
    print(f"  Samples per letter: {args.samples}")
    print("=" * 55)
    print("\n  Controls inside the window:")
    print(f"  SPACE  — Start {COUNTDOWN_SECONDS}s countdown, then record / Pause")
    print("  N      — Skip this letter")
    print("  B      — Redo this letter")
    print("  Q      — Save progress and quit\n")

    cap       = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    extractor = HandLandmarkExtractor(max_hands=1)

    completed = []
    skipped   = []

    for i, letter in enumerate(letters):
        # Skip if already collected (unless --redo)
        if not args.redo and already_collected(letter):
            print(f"  [Skip] {letter} — already in dataset")
            completed.append(letter)
            continue

        print(f"\n  [{i+1}/{len(letters)}] Collecting: {letter}")
        print(f"  → {ASL_HINTS.get(letter, '')}")

        samples = collect_letter(
            cap, extractor, letter,
            target=args.samples,
            letter_index=len(completed),
            total_letters=len(letters),
        )

        if samples is None:
            print("\n  [Quit] Saving progress...")
            break
        elif len(samples) == 0:
            skipped.append(letter)
            print(f"  [Skipped] {letter}")
        else:
            n = save_samples(samples, letter)
            completed.append(letter)
            print(f"  [Saved] {letter} — {n} rows (raw + augmented)")

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Collection complete!")
    print(f"  Collected : {len(completed)} letters — {' '.join(completed)}")
    if skipped:
        print(f"  Skipped   : {' '.join(skipped)}")
    print(f"  Dataset   : {CSV_FILE}")

    if os.path.exists(CSV_FILE):
        df   = pd.read_csv(CSV_FILE)
        dist = df["gesture"].value_counts().sort_index()
        print(f"\n  Samples per letter:")
        for ltr, cnt in dist.items():
            bar = "█" * (cnt // 30)
            print(f"    {ltr}  {bar} {cnt}")

    print("\n  Next step — train the model:")
    print("  py train_pipeline.py")
    print("=" * 55)


if __name__ == "__main__":
    main()