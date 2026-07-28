"""
collect_numbers.py - Guided ASL digit (0-9) data collection
=============================================================
Walks you through all 10 digits one by one.
Shows an on-screen reference diagram + live webcam + progress bar.

Usage:
    py collect_numbers.py                   # collect all 10 digits
    py collect_numbers.py --start 5         # resume from digit 5
    py collect_numbers.py --digit 7         # collect only digit 7
    py collect_numbers.py --samples 300     # 300 samples per digit

Writes only to data/processed/landmarks_number.csv — never touches the
alphabet dataset (landmarks_static.csv).
"""

import os
import sys
import argparse
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import NUMBER_CLASSES, LANDMARK_FEATURES, NUMBER_CSV_FILE, PROCESSED_DIR
from src.landmark_extractor import HandLandmarkExtractor
from src.dataset import augment_sample

# ─── ASL digit handshape descriptions (standard one-handed ASL counting) ─────
ASL_DIGIT_HINTS = {
    "0": "All fingers curved to meet thumb, forming a closed 'O' shape",
    "1": "Index finger straight up, other fingers + thumb folded into fist",
    "2": "Index + middle fingers up together (peace-sign shape), others folded",
    "3": "Thumb + index + middle finger extended, ring/pinky folded",
    "4": "Four fingers straight up spread apart, thumb folded across palm",
    "5": "All five fingers spread open, palm facing forward",
    "6": "Thumb touches pinky tip, other three fingers extended up",
    "7": "Thumb touches ring finger tip, other fingers extended up",
    "8": "Thumb touches middle finger tip, other fingers extended up",
    "9": "Thumb touches index finger tip, other fingers extended up",
}

# Confusable letter groups — shown as a warning (numbers vs. alphabet)
CONFUSABLE = {
    "1": ["D"],
    "2": ["V"],
    "6": ["W"],
}

# ─── Drawing helpers ──────────────────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_S     = cv2.FONT_HERSHEY_SIMPLEX
GREEN      = (0, 220, 100)
YELLOW     = (0, 220, 220)
RED        = (60,  60, 220)
WHITE      = (240, 240, 240)
PANEL_BG   = (30,  30,  40)


def draw_digit_panel(frame, digit: str, collected: int, target: int,
                     total_done: int, total: int, recording: bool):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 320, 0), (w, h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (w - 320, 0), (w - 320, h), GREEN, 2)

    px = w - 305

    cv2.putText(frame, "DIGIT", (px, 40), FONT_S, 0.6, (120, 120, 130), 1, cv2.LINE_AA)
    cv2.putText(frame, digit,  (px, 115), FONT, 3.2, GREEN, 5, cv2.LINE_AA)

    cv2.putText(frame, f"Overall: {total_done}/{total} digits",
                (px, 145), FONT_S, 0.55, (150, 150, 160), 1, cv2.LINE_AA)

    cv2.putText(frame, "HOW TO SIGN:", (px, 175), FONT_S, 0.55, YELLOW, 1, cv2.LINE_AA)
    hint = ASL_DIGIT_HINTS.get(digit, "")
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

    if digit in CONFUSABLE:
        similar = ", ".join(CONFUSABLE[digit])
        cv2.putText(frame, f"Watch out: similar to {similar}",
                    (px, 275), FONT_S, 0.48, (80, 80, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, "SAMPLES", (px, 310), FONT_S, 0.55, (120, 120, 130), 1, cv2.LINE_AA)
    bar_w = 280
    pct   = min(collected / target, 1.0)
    cv2.rectangle(frame, (px, 318), (px + bar_w, 340), (50, 50, 60), -1)
    fill_color = GREEN if pct < 0.99 else (0, 180, 255)
    cv2.rectangle(frame, (px, 318), (px + int(bar_w * pct), 340), fill_color, -1)
    cv2.putText(frame, f"{collected} / {target}",
                (px, 360), FONT_S, 0.65, WHITE, 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (px + 10, 385), 10, RED, -1)
        cv2.putText(frame, "RECORDING", (px + 28, 391),
                    FONT_S, 0.65, RED, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "PAUSED  — press SPACE",
                    (px, 391), FONT_S, 0.55, YELLOW, 1, cv2.LINE_AA)

    y = h - 130
    for key, desc in [("SPACE", "Start / Pause"), ("N", "Skip digit"),
                       ("B", "Redo digit"), ("Q", "Save & Quit")]:
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


# ─── Collection logic ─────────────────────────────────────────────────────────

def collect_digit(
    cap,
    extractor: HandLandmarkExtractor,
    digit: str,
    target: int,
    digit_index: int,
    total_digits: int,
) -> list:
    """Collect `target` landmark samples for one digit. Returns list of np.ndarray."""
    samples   = []
    recording = False

    while len(samples) < target:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        features, results = extractor.extract_features(frame)
        extractor.draw_landmarks(frame, results, color_override=(0, 200, 100))

        if features is None:
            draw_hand_missing(frame)
            recording = False

        draw_digit_panel(frame, digit, len(samples), target,
                         digit_index, total_digits, recording)

        if recording and features is not None:
            samples.append(features)
            if len(samples) % 10 == 0:
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (frame.shape[1] - 320, frame.shape[0]),
                              (0, 80, 0), -1)
                cv2.addWeighted(flash, 0.15, frame, 0.85, 0, frame)

        cv2.imshow("ASL Number Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
        elif key == ord('n'):
            print(f"  [Skip] {digit}")
            return []
        elif key == ord('b'):
            print(f"  [Redo] {digit}")
            samples = []
            recording = False
        elif key == ord('q'):
            return None

    return samples


def save_samples(samples: list, digit: str):
    """Save + augment samples to the number CSV."""
    if not samples:
        return 0

    all_rows = []
    for s in samples:
        all_rows.append(s)
        all_rows.extend(augment_sample(s))

    cols = [f"f{i}" for i in range(LANDMARK_FEATURES)]
    df   = pd.DataFrame(all_rows, columns=cols)
    df.insert(0, "gesture", digit)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    mode   = 'a' if os.path.exists(NUMBER_CSV_FILE) else 'w'
    header = not os.path.exists(NUMBER_CSV_FILE)
    df.to_csv(NUMBER_CSV_FILE, mode=mode, header=header, index=False)
    return len(all_rows)


def already_collected(digit: str) -> bool:
    if not os.path.exists(NUMBER_CSV_FILE):
        return False
    df = pd.read_csv(NUMBER_CSV_FILE)
    return digit in df["gesture"].astype(str).values


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Guided ASL digit (0-9) data collection")
    parser.add_argument("--start",   default="0",
                        help="Digit to start from (e.g. --start 5)")
    parser.add_argument("--digit",   default=None,
                        help="Collect only one specific digit")
    parser.add_argument("--samples", type=int, default=200,
                        help="Samples per digit (default: 200)")
    parser.add_argument("--redo",    action="store_true",
                        help="Re-collect digits even if already in dataset")
    args = parser.parse_args()

    if args.digit:
        digits = [args.digit]
    else:
        start_idx = NUMBER_CLASSES.index(args.start) if args.start in NUMBER_CLASSES else 0
        digits    = NUMBER_CLASSES[start_idx:]

    print("\n" + "=" * 55)
    print(f"  ASL Digit Data Collection")
    print(f"  Digits: {' '.join(digits)}")
    print(f"  Samples per digit: {args.samples}")
    print("=" * 55)
    print("\n  Controls inside the window:")
    print("  SPACE  — Start / Pause recording")
    print("  N      — Skip this digit")
    print("  B      — Redo this digit")
    print("  Q      — Save progress and quit\n")

    cap       = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    extractor = HandLandmarkExtractor(max_hands=1)

    completed = []
    skipped   = []

    for i, digit in enumerate(digits):
        if not args.redo and already_collected(digit):
            print(f"  [Skip] {digit} — already in dataset")
            completed.append(digit)
            continue

        print(f"\n  [{i+1}/{len(digits)}] Collecting: {digit}")
        print(f"  → {ASL_DIGIT_HINTS.get(digit, '')}")

        samples = collect_digit(
            cap, extractor, digit,
            target=args.samples,
            digit_index=len(completed),
            total_digits=len(digits),
        )

        if samples is None:
            print("\n  [Quit] Saving progress...")
            break
        elif len(samples) == 0:
            skipped.append(digit)
            print(f"  [Skipped] {digit}")
        else:
            n = save_samples(samples, digit)
            completed.append(digit)
            print(f"  [Saved] {digit} — {n} rows (raw + augmented)")

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print(f"  Collection complete!")
    print(f"  Collected : {len(completed)} digits — {' '.join(completed)}")
    if skipped:
        print(f"  Skipped   : {' '.join(skipped)}")
    print(f"  Dataset   : {NUMBER_CSV_FILE}")

    if os.path.exists(NUMBER_CSV_FILE):
        df   = pd.read_csv(NUMBER_CSV_FILE)
        dist = df["gesture"].astype(str).value_counts().sort_index()
        print(f"\n  Samples per digit:")
        for d, cnt in dist.items():
            bar = "█" * (cnt // 30)
            print(f"    {d}  {bar} {cnt}")

    print("\n  Next step — train the model:")
    print("  py train_numbers.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
