"""
server.py - Flask-SocketIO bridge between the ASL gesture AI pipeline and the
HandTalk AI web dashboard.

This file contains NO AI logic of its own. It re-uses exactly the same
components main.py's run_demo() uses:
    - src/landmark_extractor.py  (MediaPipe hand landmarks)
    - src/predictor.py           (StaticPredictor, used for both the
                                   alphabet model and the number model)
    - src/recognition_pipeline.py (MultiModelRecognizer, merges the
                                   alphabet + number predictions)
    - src/motion_model.py        (MotionPredictor, for J/Z)
    - src/speech.py              (SpeechOutput / TTS)

The sentence-builder (hold-to-confirm letters) and the SPEAK-gesture
hold-to-speak trigger are ported as-is from main.py's WordBuilder and
run_demo() loop. server.py's only job is to run that same loop headlessly
(no cv2.imshow) and push the results to the browser over WebSocket instead
of drawing them to an OpenCV window.

Socket events
--------------
Server -> Dashboard:
    "frame"  : {
        image,            # base64 JPEG of the current camera frame
        fps,
        gesture,          # current letter / "No Hand" / "Unknown"
        confidence,
        top_k,            # [{label, confidence}, ...]
        mode,             # "STATIC" or "HYBRID"
        sentence,         # current sentence being built
        speak_progress,   # 0..1, SPEAK-gesture hold progress
    }
    "spoken" : { text }   # confirms a sentence was actually spoken by TTS
    "server_error" : { message }

Dashboard -> Server:
    "speak"  : { text }   # user pressed the Speak button

Camera lifecycle is tied to socket connections: the camera starts on the
first client connect and stops when the last client disconnects, so it
matches the dashboard's existing Start/Stop button behavior (Start ->
connectSocket(), Stop -> socket.disconnect()).
"""

import os
import sys
import time
import base64
import threading
import traceback

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, send_from_directory, session, redirect, jsonify

from config import ENABLE_LOGIN, SHOW_PROFILE, SHOW_FAVORITES, SHOW_ANALYTICS
from flask_socketio import SocketIO

from src.config import (
    STATIC_MODEL_PATH, LABEL_ENCODER_PATH,
    MOTION_MODEL_PATH, MOTION_ENCODER_PATH,
    NUMBER_MODEL_PATH, NUMBER_ENCODER_PATH,
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS_TARGET,
    CONFIDENCE_THRESHOLD, MOTION_THRESHOLD,
    SPEAK_LABEL, SPEAK_HOLD_SECONDS, DELETE_LABEL,
)
from src.landmark_extractor import HandLandmarkExtractor
from src.predictor import StaticPredictor
from src.recognition_pipeline import MultiModelRecognizer
from src.speech import SpeechOutput, list_voices

# ─── App / Socket setup ───────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

app = Flask(__name__, static_folder=DASHBOARD_DIR, static_url_path="")
app.config["SECRET_KEY"] = "handtalk-ai"

# Remember Me (see auth/auth_routes.py's login route): when a user
# checks "Remember Me", that route sets session.permanent = True,
# which makes Flask use this lifetime instead of expiring the session
# cookie when the browser closes. Leaving "Remember Me" unchecked is
# completely unaffected - it still uses a normal, non-permanent
# session exactly as before this setting was added.
from datetime import timedelta
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

# Login system (registration/login/logout/me) - added in Phase 2.
# Lives entirely in auth/ and database/; does not touch the AI pipeline.
from auth.auth_routes import auth_bp
app.register_blueprint(auth_bp)

from history.history_routes import history_bp
app.register_blueprint(history_bp)
from history.history_service import save_sentence

from favorites.favorites_routes import favorites_bp
app.register_blueprint(favorites_bp)

from user_profile.profile_routes import profile_bp
app.register_blueprint(profile_bp)

from settings.settings_routes import settings_bp
app.register_blueprint(settings_bp)
from auth.auth_service import touch_last_active

from feedback.feedback_routes import feedback_bp
app.register_blueprint(feedback_bp)

# "threading" async mode avoids eventlet/gevent monkey-patching, which can
# clash with OpenCV/TensorFlow's own native threads.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

CAMERA_INDEX = 0
JPEG_QUALITY = 70


@app.route("/")
def landing():
    # New route: serves the public landing page. Added on top of the
    # existing app - does not touch auth, the AI pipeline, or any
    # other route below.
    return send_from_directory(DASHBOARD_DIR, "landing.html")


@app.route("/dashboard")
def index():
    if not ENABLE_LOGIN or "user_id" in session:
        return send_from_directory(DASHBOARD_DIR, "index.html")
    return redirect("/login.html")


@app.route("/login.html")
def login_page():
    if not ENABLE_LOGIN:
        return redirect("/dashboard")
    return send_from_directory(DASHBOARD_DIR, "login.html")


@app.route("/register.html")
def register_page():
    if not ENABLE_LOGIN:
        return redirect("/dashboard")
    return send_from_directory(DASHBOARD_DIR, "register.html")


@app.route("/api/config")
def get_config():
    return jsonify({
        "enable_login":    ENABLE_LOGIN,
        "show_profile":    SHOW_PROFILE,
        "show_favorites":  SHOW_FAVORITES,
        "show_analytics":  SHOW_ANALYTICS,
    })


@app.route("/api/voices")
def get_voices():
    # Not tied to the camera/GestureStreamer - just enumerates installed
    # system voices, so it works whether or not the socket is connected
    # (i.e. the Settings page can populate this without "Start Feed").
    return jsonify({"voices": list_voices()})


# ─── Sentence builder (ported as-is from main.py's WordBuilder) ────────────
class WordBuilder:
    def __init__(self, hold=20):
        self.sentence = ""
        self._last = ""
        self._cnt = 0
        self._hold = hold

    def update(self, gesture, conf):
        if not gesture or gesture in ("Unknown", "No Hand", "—", SPEAK_LABEL):
            self._last = ""
            self._cnt = 0
            return False
        if gesture == self._last:
            self._cnt += 1
        else:
            self._last = gesture
            self._cnt = 1
            return False
        if self._cnt == self._hold:
            if gesture == DELETE_LABEL:
                # Delete the last character — same behaviour as SPACE but removes
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
        if self.sentence:
            self.sentence = self.sentence[:-1]

    def clear(self):
        self.sentence = ""
        self._last = ""
        self._cnt = 0


# ─── Camera / inference worker ──────────────────────────────────────────────
class GestureStreamer:
    """
    Owns the camera and the AI pipeline. Runs the same loop as main.py's
    run_demo(), minus cv2.imshow, and emits a 'frame' SocketIO event per
    processed frame instead.
    """

    def __init__(self):
        self.static_ok = os.path.exists(STATIC_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH)
        self.motion_ok = os.path.exists(MOTION_MODEL_PATH) and os.path.exists(MOTION_ENCODER_PATH)

        if not self.static_ok:
            raise RuntimeError(
                "No static model found. Train it first: py train_pipeline.py"
            )

        self.static_pred = StaticPredictor(
            STATIC_MODEL_PATH, LABEL_ENCODER_PATH, threshold=CONFIDENCE_THRESHOLD
        )

        # Modular recognition pipeline: the alphabet model is always
        # registered; the number model joins in if it has been trained.
        # Future models (motion-as-registrable, common words, ...) plug in
        # the same way, with no change to the merge logic itself.
        self.number_ok = os.path.exists(NUMBER_MODEL_PATH) and os.path.exists(NUMBER_ENCODER_PATH)
        self.recognizer = MultiModelRecognizer()
        self.recognizer.register("alphabet", self.static_pred)
        if self.number_ok:
            self.number_pred = StaticPredictor(
                NUMBER_MODEL_PATH, NUMBER_ENCODER_PATH, threshold=CONFIDENCE_THRESHOLD
            )
            self.recognizer.register("number", self.number_pred)

        self.motion_pred = None
        if self.motion_ok:
            from src.motion_model import MotionPredictor
            self.motion_pred = MotionPredictor(
                MOTION_MODEL_PATH, MOTION_ENCODER_PATH, threshold=MOTION_THRESHOLD
            )

        self.extractor = HandLandmarkExtractor(max_hands=1)
        self.speech    = SpeechOutput()
        self.builder   = WordBuilder(hold=20)

        self._clahe     = cv2.createCLAHE(2.0, (8, 8))
        self._cap       = None
        self._thread    = None
        self._running   = False
        self._lock      = threading.Lock()

        self._motion_label = None
        self._motion_conf  = 0.0
        self._motion_display_frames = 0

        self._speak_start = None

        # Which logged-in user (and login session row) the camera is
        # currently running for. Set in on_connect() since the camera
        # loop itself runs in a background thread with no Flask session
        # access. Used to attribute saved sentences to the right user.
        self.active_user_id    = None
        self.active_session_id = None
        self.show_landmarks    = True   # toggled by "set_landmarks" socket event

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        with self._lock:
            if self._running:
                return
            print(f"[server] Opening camera index {CAMERA_INDEX} ...")
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                msg = (
                    f"Could not open camera index {CAMERA_INDEX}. "
                    "It may be in use by another app (close Camera app / Zoom / "
                    "the old browser tab), blocked by Windows camera privacy "
                    "settings, or you may need a different CAMERA_INDEX."
                )
                print(f"[server] {msg}")
                socketio.emit("server_error", {"message": msg})
                raise RuntimeError(msg)

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
            print("[server] Camera opened successfully.")
            self.builder.clear()
            if self.motion_pred:
                self.motion_pred.reset()
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None

    def speak_text(self, text):
        text = (text or "").strip()
        if not text:
            return
        self.speech.speak(text)
        socketio.emit("spoken", {"text": text})

        if self.active_user_id:
            try:
                save_sentence(
                    user_id=self.active_user_id,
                    sentence=text,
                    session_id=self.active_session_id,
                )
            except Exception as e:
                # Never let a database hiccup break speech or the camera loop.
                print(f"[server] Failed to save sentence history: {e}")

    def clear_sentence(self):
        # WordBuilder is the single source of truth for the sentence - the
        # dashboard's Clear button only clears the browser-side copy, so
        # without this the next 'frame' event (still coming from the AI
        # loop while the camera is running) would overwrite it right back.
        self.builder.clear()

    # -- main loop --------------------------------------------------------------
    def _loop(self):
        fps_c, fps_t, fps_d = 0, time.time(), 0.0
        gesture, confidence, top_k = "—", 0.0, []
        speak_progress = 0.0

        try:
            consecutive_failures = 0
            while self._running:
                frame_start = time.time()
                ret, frame = self._cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        print("[server] Camera read failed, retrying...")
                    if consecutive_failures >= 60:  # ~3s of failures at 0.05s sleep
                        msg = (
                            f"Camera stopped returning frames after "
                            f"{consecutive_failures} failed reads. It may have "
                            "been disconnected or grabbed by another app."
                        )
                        print(f"[server] {msg}")
                        socketio.emit("server_error", {"message": msg})
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
                frame = cv2.flip(frame, 1)
                enh = self._enhance(frame)

                features, results = self.extractor.extract_features(enh)
                if self.show_landmarks:
                    self.extractor.draw_landmarks(frame, results)

                if features is not None:
                    if self.motion_pred:
                        self.motion_pred.push(features)
                        ml, mc = self.motion_pred.predict()
                        if ml is not None:
                            self._motion_label, self._motion_conf = ml, mc
                            self._motion_display_frames = 30

                    s_label, s_conf, top_k, _source = self.recognizer.predict_with_top_k(features, k=3)

                    if self._motion_display_frames > 0:
                        gesture, confidence = self._motion_label, self._motion_conf
                        self._motion_display_frames -= 1
                    else:
                        gesture, confidence = s_label, s_conf
                else:
                    gesture, confidence, top_k = "No Hand", 0.0, []
                    if self.motion_pred:
                        self.motion_pred.reset()
                    self.builder._last = ""
                    self.builder._cnt = 0

                # SPEAK (thumbs-up hold) trigger
                speak_now = gesture == SPEAK_LABEL and confidence >= CONFIDENCE_THRESHOLD
                if speak_now and self.builder.sentence.strip():
                    if self._speak_start is None:
                        self._speak_start = time.time()
                    elapsed = time.time() - self._speak_start
                    speak_progress = min(elapsed / SPEAK_HOLD_SECONDS, 1.0)
                    if elapsed >= SPEAK_HOLD_SECONDS:
                        sentence_to_speak = self.builder.sentence.strip()
                        self.speak_text(sentence_to_speak)
                        self.builder.clear()
                        self._speak_start = None
                        speak_progress = 0.0
                else:
                    self._speak_start = None
                    speak_progress = 0.0

                if speak_progress < 0.1:
                    self.builder.update(gesture, confidence)

                # FPS
                fps_c += 1
                if time.time() - fps_t >= 1.0:
                    fps_d = fps_c / (time.time() - fps_t)
                    fps_c, fps_t = 0, time.time()

                # Encode + emit
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    image_b64 = base64.b64encode(buf).decode("utf-8")
                    socketio.emit("frame", {
                        "image":          image_b64,
                        "fps":            fps_d,
                        "gesture":        gesture,
                        "confidence":     confidence,
                        "top_k":          [{"label": l, "confidence": c} for l, c in top_k],
                        "mode":           "HYBRID" if self.motion_ok else "STATIC",
                        "sentence":       self.builder.sentence,
                        "speak_progress": speak_progress,
                    })

                elapsed = time.time() - frame_start
                socketio.sleep(max(0.0, 1.0 / FPS_TARGET - elapsed))
        except Exception:
            traceback.print_exc()
            socketio.emit("server_error", {"message": "Camera loop crashed. Check server logs."})
        finally:
            self._running = False

    def _enhance(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─── Streamer singleton + connected-client tracking ────────────────────────
streamer = None
_connected_clients = 0
_clients_lock = threading.Lock()


def get_streamer():
    global streamer
    if streamer is None:
        streamer = GestureStreamer()
    return streamer


# ─── Socket events ──────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    global _connected_clients
    if not ENABLE_LOGIN:
        with _clients_lock:
            _connected_clients += 1
            count = _connected_clients
        print(f"[server] Client connected (guest mode, {count} active)")
        try:
            gs = get_streamer()
            gs.active_user_id    = None
            gs.active_session_id = None
            gs.start()
        except Exception as e:
            print(f"[server] Failed to start camera/AI pipeline: {e}")
            socketio.emit("server_error", {"message": str(e)})
        return

    if "user_id" not in session:
        print("[server] Rejected socket connection - no active login session.")
        return False  # refuses the connection
    with _clients_lock:
        _connected_clients += 1
        count = _connected_clients
    print(f"[server] Client connected ({count} active)")
    try:
        gs = get_streamer()
        gs.active_user_id    = session["user_id"]
        gs.active_session_id = session.get("session_id")
        gs.start()
        touch_last_active(session["user_id"])
    except Exception as e:
        print(f"[server] Failed to start camera/AI pipeline: {e}")
        socketio.emit("server_error", {"message": str(e)})


@socketio.on("disconnect")
def on_disconnect():
    global _connected_clients
    with _clients_lock:
        _connected_clients = max(0, _connected_clients - 1)
        count = _connected_clients
    print(f"[server] Client disconnected ({count} active)")
    if count == 0 and streamer is not None:
        streamer.stop()


@socketio.on("speak")
def on_speak(data):
    text = (data or {}).get("text", "")
    try:
        get_streamer().speak_text(text)
    except Exception as e:
        socketio.emit("server_error", {"message": str(e)})


@socketio.on("clear_sentence")
def on_clear_sentence():
    try:
        get_streamer().clear_sentence()
    except Exception as e:
        socketio.emit("server_error", {"message": str(e)})


@socketio.on("set_landmarks")
def on_set_landmarks(data):
    try:
        get_streamer().show_landmarks = bool((data or {}).get("visible", True))
    except Exception:
        pass


@socketio.on("set_speech")
def on_set_speech(data):
    """Applies the Settings page's Speech Speed / Speech Volume / Speech
    Voice to the actual TTS voice (see src/speech.py's SpeechOutput
    set_speed/set_volume/set_voice), so both the SPEAK-gesture hold and the
    manual Speak button use them."""
    try:
        data = data or {}
        speech = get_streamer().speech
        if "speed" in data:
            speech.set_speed(data["speed"])
        if "volume" in data:
            speech.set_volume(data["volume"])
        if "voice" in data:
            speech.set_voice(data["voice"])
    except Exception:
        pass


# ─── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[server] HandTalk AI bridge starting at http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000)