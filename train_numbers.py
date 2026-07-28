"""
train_numbers.py - Trains the Number (digit 0-9) model ONLY.
Never imports GestureTrainer, train_static_model, or train_motion_model,
so it cannot touch the alphabet or motion model/dataset by construction.

Usage:
    py train_numbers.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from src.config import NUMBER_CSV_FILE

    if not os.path.exists(NUMBER_CSV_FILE):
        print("[Pipeline] No number dataset found!")
        print("  Run:  py collect_numbers.py  (for digits 0-9)")
        return

    from src.number_model import train_number_model
    print("[Pipeline] Training number model (10-class) ...")
    _, _, acc = train_number_model()
    print(f"✓ Number model accuracy: {acc:.4f}")
    print("\n[Pipeline] Done!  Run:  py main.py demo --tts")


if __name__ == "__main__":
    main()
