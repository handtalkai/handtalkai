"""
main.py - Hybrid ASL gesture recognition (static + LSTM motion)
Sentence TTS: hold THUMBS UP (SPEAK gesture) for 1.5 seconds to speak full sentence.
No fist detection — SPEAK is a trained gesture class.
"""
import os, sys, time, argparse
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    GESTURE_CLASSES, STATIC_LETTERS, MOTION_LETTERS,
    STATIC_MODEL_PATH, LABEL_ENCODER_PATH,
    MOTION_MODEL_PATH, MOTION_ENCODER_PATH,
    NUMBER_MODEL_PATH, NUMBER_ENCODER_PATH,
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS_TARGET,
    CONFIDENCE_THRESHOLD, MOTION_THRESHOLD,
    SPEAK_LABEL, SPEAK_HOLD_SECONDS, DELETE_LABEL,
)
from src.landmark_extractor import HandLandmarkExtractor
from src.recognition_pipeline import MultiModelRecognizer
from src.ui import draw_gesture_panel, draw_status_bar, draw_help_overlay, PALETTE
from src.speech import SpeechOutput

FONT_S = cv2.FONT_HERSHEY_SIMPLEX
FONT   = cv2.FONT_HERSHEY_DUPLEX


# ─── Camera ──────────────────────────────────────────────────────────────────

def _open_cam(idx=0):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    return cap


_CLAHE = cv2.createCLAHE(2.0, (8, 8))

def _enhance(frame):
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l      = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─── Word / sentence builder ─────────────────────────────────────────────────

class WordBuilder:
    def __init__(self, hold=20):
        self.sentence = ""
        self._last = ""; self._cnt = 0; self._hold = hold

    def update(self, gesture, conf):
        # Ignore trigger-only gestures — not letters
        if not gesture or gesture in ("Unknown", "No Hand", "—", SPEAK_LABEL):
            self._last = ""; self._cnt = 0; return False
        if gesture == self._last:
            self._cnt += 1
        else:
            self._last = gesture; self._cnt = 1; return False
        if self._cnt == self._hold:
            if gesture == DELETE_LABEL:
                # Delete the last character from the sentence
                if self.sentence:
                    self.sentence = self.sentence[:-1]
            elif gesture == "SPACE":
                if self.sentence and self.sentence[-1] != " ":
                    self.sentence += " "
            else:
                self.sentence += gesture
            return True
        return False

    @property
    def progress(self):
        return min(self._cnt / self._hold, 1.0)

    def backspace(self):
        if self.sentence: self.sentence = self.sentence[:-1]

    def clear(self):
        self.sentence = ""; self._last = ""; self._cnt = 0


# ─── Draw helpers ─────────────────────────────────────────────────────────────

def draw_builder(frame, builder, gesture, speak_progress=0.0):
    h, w = frame.shape[:2]
    pw   = w - 320
    ov   = frame.copy()
    cv2.rectangle(ov, (0, h-70), (pw, h), (20, 20, 28), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, h-70), (pw, h-70), PALETTE["accent"], 2)

    cv2.putText(frame, "SENTENCE", (10, h-53),
                FONT_S, 0.5, (100, 100, 110), 1, cv2.LINE_AA)
    disp = builder.sentence + ("_" if gesture not in ("Unknown","No Hand","—","",SPEAK_LABEL,DELETE_LABEL) else "")
    if len(disp) > 55: disp = "…" + disp[-54:]
    cv2.putText(frame, disp, (10, h-22), FONT, 0.85, (240, 240, 240), 2, cv2.LINE_AA)

    # Letter hold progress bar
    if builder.progress > 0 and gesture not in ("Unknown","No Hand","—","",SPEAK_LABEL,DELETE_LABEL):
        bw = int((pw - 20) * builder.progress)
        cv2.rectangle(frame, (10, h-7), (10+bw, h-1), PALETTE["accent"], -1)
        cv2.putText(frame, f"Hold: {gesture}", (12, h-9),
                    FONT_S, 0.42, (10, 10, 10), 1, cv2.LINE_AA)

    # SPEAK gesture countdown ring
    if speak_progress > 0:
        cx, cy = pw - 50, h - 35
        radius = 28
        cv2.circle(frame, (cx, cy), radius, (50, 50, 60), -1)
        cv2.circle(frame, (cx, cy), radius, (80, 80, 90), 2)
        angle = int(360 * speak_progress)
        for deg in range(0, angle, 3):
            rad = np.deg2rad(deg - 90)
            x   = int(cx + radius * np.cos(rad))
            y   = int(cy + radius * np.sin(rad))
            cv2.circle(frame, (x, y), 3, (0, 200, 255), -1)
        pct_txt = f"{int(speak_progress*100)}%"
        cv2.putText(frame, pct_txt,    (cx-14, cy+5),  FONT_S, 0.45, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(frame, "SPEAK",    (cx-20, cy-40), FONT_S, 0.45, (0,200,255),   1, cv2.LINE_AA)
        cv2.putText(frame, "gesture",  (cx-22, cy+42), FONT_S, 0.40, (0,200,255),   1, cv2.LINE_AA)

    cv2.putText(frame, "[C] Clear  [Bksp] Delete  | Thumbs Up to Speak",
                (10, h-55), FONT_S, 0.40, (70, 70, 80), 1, cv2.LINE_AA)


def draw_model_badge(frame, static_ok, motion_ok):
    y = 50
    for label, ok in [("STATIC", static_ok), ("J/Z LSTM", motion_ok)]:
        color = (0, 200, 80) if ok else (60, 60, 180)
        cv2.putText(frame, f"[{label}{'OK' if ok else 'X'}]",
                    (10, y), FONT_S, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def draw_speaking_flash(frame, sentence):
    """Green flash overlay when sentence is spoken."""
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 60, 0), -1)
    cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, "SPEAKING", (w//2 - 100, h//2 - 20),
                FONT, 1.0, (0, 255, 120), 2, cv2.LINE_AA)
    # Show sentence
    words = sentence.strip().split()
    line  = ""; lines = []
    for word in words:
        if len(line) + len(word) + 1 <= 40:
            line += (" " if line else "") + word
        else:
            lines.append(line); line = word
    if line: lines.append(line)
    for i, l in enumerate(lines[:3]):
        cv2.putText(frame, l, (w//2 - len(l)*8, h//2 + 20 + i*30),
                    FONT_S, 0.7, (200, 255, 200), 1, cv2.LINE_AA)


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo(camera=0, tts=False):
    static_ok = os.path.exists(STATIC_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH)
    motion_ok = os.path.exists(MOTION_MODEL_PATH) and os.path.exists(MOTION_ENCODER_PATH)

    if not static_ok:
        print("[Demo] No static model found. Run: py train_pipeline.py")
        sys.exit(1)

    from src.predictor import StaticPredictor
    static_pred = StaticPredictor(STATIC_MODEL_PATH, LABEL_ENCODER_PATH,
                                  threshold=CONFIDENCE_THRESHOLD)

    # Modular recognition pipeline: alphabet is always registered; the
    # number model joins in if it has been trained. Future models plug in
    # the same way, with no change to the merge logic itself.
    number_ok = os.path.exists(NUMBER_MODEL_PATH) and os.path.exists(NUMBER_ENCODER_PATH)
    recognizer = MultiModelRecognizer()
    recognizer.register("alphabet", static_pred)
    if number_ok:
        number_pred = StaticPredictor(NUMBER_MODEL_PATH, NUMBER_ENCODER_PATH,
                                      threshold=CONFIDENCE_THRESHOLD)
        recognizer.register("number", number_pred)
        print("[Demo] Number model loaded — digits 0-9 enabled")

    motion_pred = None
    if motion_ok:
        from src.motion_model import MotionPredictor
        motion_pred = MotionPredictor(MOTION_MODEL_PATH, MOTION_ENCODER_PATH,
                                      threshold=MOTION_THRESHOLD)
        print("[Demo] Hybrid mode: static + J/Z LSTM")
    else:
        print("[Demo] Static-only mode (no J/Z model found)")
        print("       Run: py collect_motion.py  then  py train_pipeline.py")

    speech    = SpeechOutput() if tts else None
    builder   = WordBuilder(hold=20)
    extractor = HandLandmarkExtractor(max_hands=1)
    cap       = _open_cam(camera)

    show_skeleton  = True
    show_help      = False
    live           = True
    fps_c = 0; fps_d = 0.0; fps_t = time.time()
    gesture = "—"; confidence = 0.0; top_k = []
    status  = "Sign letters → Thumbs Up 1.5s to speak  |  [H] help  [Q] quit"

    motion_label = None; motion_conf = 0.0; motion_display_frames = 0
    _miss_count  = 0    # consecutive frames with no hand detected

    # SPEAK trigger state
    speak_start    = None
    speak_progress = 0.0
    speaking_flash = 0
    last_spoken    = ""

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        enh   = _enhance(frame)

        features, results = extractor.extract_features(enh)
        if show_skeleton:
            extractor.draw_landmarks(frame, results, color_override=PALETTE["skeleton"])

        # ── Gesture inference ─────────────────────────────────────────────────
        if live and features is not None:
            _miss_count = 0   # hand visible — reset miss counter

            if motion_pred:
                motion_pred.push(features)
                ml, mc = motion_pred.predict()
                if ml is not None:
                    motion_label = ml; motion_conf = mc
                    motion_display_frames = 30
                    # Clear static smoother so stale votes don't override
                    static_pred.smoother.reset()

            if motion_display_frames > 0:
                # Motion result active — skip static so smoother stays clean
                gesture    = motion_label
                confidence = motion_conf
                motion_display_frames -= 1
                top_k = [(motion_label, motion_conf)]
            else:
                s_label, s_conf, top_k, _source = recognizer.predict_with_top_k(features, k=3)
                gesture    = s_label
                confidence = s_conf

        elif features is None:
            gesture = "No Hand"; confidence = 0.0; top_k = []
            _miss_count += 1
            # Reset motion buffer only after 5 consecutive missed frames
            if motion_pred and _miss_count >= 5:
                motion_pred.reset()
            builder._last = ""; builder._cnt = 0

        # ── SPEAK gesture trigger (thumbs up) ─────────────────────────────────
        speak_now = (gesture == SPEAK_LABEL and confidence >= CONFIDENCE_THRESHOLD)

        if speak_now and builder.sentence.strip():
            if speak_start is None:
                speak_start = time.time()
            elapsed        = time.time() - speak_start
            speak_progress = min(elapsed / SPEAK_HOLD_SECONDS, 1.0)

            if elapsed >= SPEAK_HOLD_SECONDS:
                # Speak the full sentence
                sentence_to_speak = builder.sentence.strip()
                print(f"[TTS] Speaking: '{sentence_to_speak}'")
                if speech:
                    speech.speak(sentence_to_speak)
                speaking_flash = 45
                last_spoken    = sentence_to_speak
                builder.clear()
                speak_start    = None
                speak_progress = 0.0
                status = f"Spoke: '{sentence_to_speak}'"
        else:
            speak_start    = None
            speak_progress = 0.0

        # Don't add letters while SPEAK gesture is held
        if speak_progress < 0.1:
            builder.update(gesture, confidence)

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_c += 1
        if time.time() - fps_t >= 1.0:
            fps_d = fps_c / (time.time() - fps_t)
            fps_c = 0; fps_t = time.time()

        # ── Draw ──────────────────────────────────────────────────────────────
        if speaking_flash > 0:
            draw_speaking_flash(frame, last_spoken)
            speaking_flash -= 1
        else:
            draw_gesture_panel(frame, gesture, confidence,
                               mode="HYBRID" if motion_ok else "STATIC",
                               fps=fps_d, top_k=top_k)

        draw_builder(frame, builder, gesture, speak_progress=speak_progress)
        draw_status_bar(frame, status, ok=True)
        draw_model_badge(frame, static_ok, motion_ok)
        if show_help: draw_help_overlay(frame)

        # J/Z buffer progress bar
        if motion_pred and motion_ok:
            buf_len = len(motion_pred._buf)
            buf_max = motion_pred.seq_len
            buf_pct = buf_len / buf_max
            fh, fw  = frame.shape[:2]
            bx, by, bw, bh = 10, fh - 110, 200, 14
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40,40,50), -1)
            bc = (0,180,255) if buf_pct < 1.0 else (0,255,100)
            cv2.rectangle(frame, (bx, by), (bx+int(bw*buf_pct), by+bh), bc, -1)
            cv2.putText(frame, f"J/Z buffer {buf_len}/{buf_max}",
                        (bx, by-5), FONT_S, 0.45, (150,150,160), 1, cv2.LINE_AA)

        cv2.imshow("ASL Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('l'): live = not live; status = f"Inference {'ON' if live else 'OFF'}"
        elif key == ord('s'): show_skeleton = not show_skeleton
        elif key == ord('h'): show_help = not show_help
        elif key == ord('c'): builder.clear(); status = "Cleared"
        elif key == 8:        builder.backspace()
        elif key == 13:       # ENTER as backup trigger
            if builder.sentence.strip():
                s = builder.sentence.strip()
                print(f"[TTS] Speaking (ENTER): '{s}'")
                if speech: speech.speak(s)
                speaking_flash = 45; last_spoken = s
                builder.clear(); status = f"Spoke: '{s}'"

    cap.release(); extractor.close()
    if speech: speech.stop()
    cv2.destroyAllWindows()
    if builder.sentence.strip():
        print(f"[Demo] Final sentence: {builder.sentence.strip()}")


# ─── Collect ──────────────────────────────────────────────────────────────────

def run_collect(gesture, samples=200):
    from src.dataset import DatasetRecorder
    if gesture not in GESTURE_CLASSES:
        print(f"Unknown gesture. Valid: {GESTURE_CLASSES}"); sys.exit(1)
    cap = _open_cam()
    rec = DatasetRecorder(gesture=gesture, target_samples=samples)
    n   = rec.record(cap)
    cap.release(); cv2.destroyAllWindows()
    if n > 0: rec.save_to_csv()


# ─── Train ────────────────────────────────────────────────────────────────────

def run_train(motion_only=False, static_only=False):
    from src.trainer import GestureTrainer
    if not motion_only:
        t = GestureTrainer(); t.print_dataset_stats()
        r = t.train_static()
        print(f"Static accuracy: {r['accuracy']:.4f}")
    if not static_only:
        from src.config import MOTION_CSV_FILE
        if os.path.exists(MOTION_CSV_FILE):
            from src.motion_model import train_motion_model
            _, _, acc = train_motion_model()
            print(f"Motion accuracy: {acc:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASL Gesture Recognition")
    sub    = parser.add_subparsers(dest="cmd")

    pc = sub.add_parser("collect")
    pc.add_argument("gesture", choices=GESTURE_CLASSES)
    pc.add_argument("--samples", type=int, default=200)

    pt = sub.add_parser("train")
    pt.add_argument("--motion-only", action="store_true")
    pt.add_argument("--static-only", action="store_true")

    pd_ = sub.add_parser("demo")
    pd_.add_argument("--camera", type=int, default=0)
    pd_.add_argument("--tts",    action="store_true")

    args = parser.parse_args()

    if   args.cmd == "collect": run_collect(args.gesture, args.samples)
    elif args.cmd == "train":   run_train(args.motion_only, args.static_only)
    elif args.cmd == "demo":    run_demo(args.camera, args.tts)
    else:
        parser.print_help()
        print("\n── Quick start ─────────────────────────────────────────────")
        print("  1. py collect_asl.py                # A-Z + SPACE")
        print("  2. py main.py collect SPEAK          # thumbs up gesture")
        print("  3. py collect_motion.py              # J and Z")
        print("  4. py train_pipeline.py              # train all models")
        print("  5. py main.py demo --tts             # run with speech")

if __name__ == "__main__":
    main()