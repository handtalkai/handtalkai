"""
train_pipeline.py - Trains BOTH models:
  1. Static dense model  (A-I, K-Y, SPACE  — 25 classes)
  2. Motion LSTM model   (J, Z             —  2 classes)
Usage:
    py train_pipeline.py
    py train_pipeline.py --static-only
    py train_pipeline.py --motion-only
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--motion-only", action="store_true")
    args = parser.parse_args()

    from src.config import CSV_FILE, MOTION_CSV_FILE, STATIC_LETTERS, MOTION_LETTERS

    # ── Static model ──────────────────────────────────────────────────────────
    if not args.motion_only:
        if not os.path.exists(CSV_FILE):
            print("[Pipeline] No static dataset found!")
            print("  Run:  py collect_asl.py  (for A-I, K-Y, SPACE)")
            if not args.static_only: pass
        else:
            from src.trainer import GestureTrainer
            trainer = GestureTrainer()
            trainer.print_dataset_stats()
            print("\n[Pipeline] Training static model (25-class) ...")
            results = trainer.train_static()
            print(f"✓ Static model accuracy: {results['accuracy']:.4f}")

    # ── Motion model ──────────────────────────────────────────────────────────
    if not args.static_only:
        if not os.path.exists(MOTION_CSV_FILE):
            print("\n[Pipeline] No motion dataset found — skipping J/Z model.")
            print("  Run:  py collect_motion.py  (for J and Z)")
        else:
            from src.motion_model import train_motion_model
            print("\n[Pipeline] Training motion model (J & Z LSTM) ...")
            _, _, acc = train_motion_model()
            print(f"✓ Motion model accuracy: {acc:.4f}")

    print("\n[Pipeline] Done!  Run:  py main.py demo --tts")

if __name__ == "__main__":
    main()