"""
clean_static_dataset.py
=======================
One-time script: removes any J and Z rows from landmarks_static.csv.

J and Z are motion gestures — single-frame snapshots of those letters
in the static dataset teach the static model to fire J/Z constantly
(whenever the hand is in the J/Z starting pose), overriding the motion
model's predictions.

Run ONCE before retraining:
    python clean_static_dataset.py

Then retrain the static model:
    py train_pipeline.py --static-only
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CSV_FILE, STATIC_CLASSES, MOTION_LETTERS


def main():
    if not os.path.exists(CSV_FILE):
        print(f"[Clean] Dataset not found: {CSV_FILE}")
        print("        Nothing to clean.")
        return

    df = pd.read_csv(CSV_FILE)
    total_before = len(df)

    # Report what's currently in the file
    print(f"[Clean] Loaded {total_before} rows from {CSV_FILE}")
    present = df["gesture"].value_counts().sort_index()
    print("[Clean] Current class distribution:")
    for gesture, count in present.items():
        marker = "  ← WILL BE REMOVED (motion gesture)" if gesture in MOTION_LETTERS else ""
        print(f"         {gesture:6s}  {count:6d}{marker}")

    # Filter out motion letters
    motion_rows = df[df["gesture"].isin(MOTION_LETTERS)]
    if len(motion_rows) == 0:
        print("\n[Clean] No J or Z rows found — dataset is already clean.")
        return

    df_clean = df[df["gesture"].isin(STATIC_CLASSES)].copy()
    total_after  = len(df_clean)
    removed      = total_before - total_after

    # Verify all expected static classes are still present
    missing = [c for c in STATIC_CLASSES if c not in df_clean["gesture"].values]
    if missing:
        print(f"\n[Clean] WARNING: these static classes have no samples after cleaning: {missing}")
        print("        You may need to re-collect data for those letters.")

    # Write cleaned file (overwrite)
    df_clean.to_csv(CSV_FILE, index=False)

    print(f"\n[Clean] Removed {removed} rows ({', '.join(MOTION_LETTERS)} gestures).")
    print(f"[Clean] Kept    {total_after} rows across {df_clean['gesture'].nunique()} classes.")
    print(f"[Clean] Saved   → {CSV_FILE}")
    print()
    print("[Clean] Next step — retrain the static model:")
    print("        py train_pipeline.py --static-only")


if __name__ == "__main__":
    main()