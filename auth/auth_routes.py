"""
auth/auth_routes.py - Flask Blueprint exposing the login system as a small
JSON API: POST /api/register, POST /api/login, POST /api/logout, GET /api/me,
PUT /api/me, POST /api/me/password, POST /api/me/avatar.

Registered onto the existing Flask app in server.py (two lines added
there - see the comment in that file). This blueprint does not touch the
AI pipeline, the camera, or the SocketIO frame-streaming loop in any way.
"""

import os
import time
from functools import wraps

from flask import Blueprint, request, jsonify, session

from config import ENABLE_LOGIN
from werkzeug.utils import secure_filename

from auth.auth_service import (
    create_user, find_user_by_login, verify_password, get_user_by_id,
    start_session, end_session, touch_last_active,
    update_profile, update_password, update_avatar, remove_avatar,
)
from auth.password_reset_service import (
    request_password_reset, verify_reset_code, reset_password,
)

auth_bp = Blueprint("auth", __name__, url_prefix="/api")

# ── Avatar upload settings ───────────────────────────────────────────────
# dashboard/ is already served at the site root by server.py's Flask app
# (static_folder=DASHBOARD_DIR, static_url_path=""), so anything saved
# under dashboard/uploads/avatars/ is automatically reachable at
# /uploads/avatars/<filename> with no extra route needed.
_AUTH_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR  = os.path.dirname(_AUTH_DIR)
AVATAR_DIR = os.path.join(_BASE_DIR, "dashboard", "uploads", "avatars")
ALLOWED_AVATAR_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_AVATAR_BYTES    = 5 * 1024 * 1024  # 5 MB


def login_required(view_func):
    """Decorator for routes that require an active logged-in session."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not ENABLE_LOGIN:
            # Set a None sentinel so view functions can do session["user_id"]
            # without KeyError; DB reads with NULL user_id return empty results.
            if "user_id" not in session:
                session["user_id"] = None
            return view_func(*args, **kwargs)
        if "user_id" not in session:
            return jsonify({"error": "Not logged in."}), 401
        return view_func(*args, **kwargs)
    return wrapped


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    try:
        user = create_user(
            fullname=data.get("fullname"),
            email=data.get("email"),
            username=data.get("username"),
            password=data.get("password"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"message": "Registered successfully.", "user": user}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    identifier = data.get("identifier") or data.get("username") or data.get("email")
    password   = data.get("password") or ""
    remember_me = bool(data.get("remember_me"))

    user = find_user_by_login(identifier)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid username/email or password."}), 401

    # Remember Me: a "permanent" Flask session uses
    # PERMANENT_SESSION_LIFETIME (configured in server.py) instead of
    # expiring when the browser closes, so the user stays logged in
    # across browser restarts. Leaving this unchecked keeps today's
    # exact behavior - a normal, browser-session-only cookie.
    session.permanent = remember_me

    session["user_id"] = user["id"]
    session["session_id"] = start_session(user["id"])
    touch_last_active(user["id"])
    return jsonify({
        "message": "Logged in successfully.",
        "user": {
            "id":       user["id"],
            "fullname": user["fullname"],
            "email":    user["email"],
            "username": user["username"],
        },
    }), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    end_session(session.get("session_id"))
    session.pop("user_id", None)
    session.pop("session_id", None)
    return jsonify({"message": "Logged out."}), 200


@auth_bp.route("/me", methods=["GET"])
@login_required
def me():
    if not ENABLE_LOGIN:
        return jsonify({"user": {
            "id":          None,
            "fullname":    "Guest",
            "username":    "guest",
            "email":       "",
            "avatar_path": None,
            "created_at":  None,
            "last_active": None,
        }}), 200
    user = get_user_by_id(session["user_id"])
    if not user:
        session.pop("user_id", None)
        return jsonify({"error": "Not logged in."}), 401
    return jsonify({"user": user}), 200


@auth_bp.route("/me", methods=["PUT"])
@login_required
def update_me():
    data = request.get_json(silent=True) or {}
    try:
        user = update_profile(
            user_id=session["user_id"],
            fullname=data.get("fullname"),
            username=data.get("username"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "Profile updated.", "user": user}), 200


@auth_bp.route("/me/password", methods=["POST"])
@login_required
def change_password():
    data = request.get_json(silent=True) or {}
    try:
        update_password(
            user_id=session["user_id"],
            current_password=data.get("current_password"),
            new_password=data.get("new_password"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "Password changed."}), 200


@auth_bp.route("/me/avatar", methods=["POST"])
@login_required
def upload_avatar():
    file = request.files.get("avatar")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_AVATAR_EXTS:
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, GIF, or WEBP."}), 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_AVATAR_BYTES:
        return jsonify({"error": "Image must be smaller than 5MB."}), 400

    os.makedirs(AVATAR_DIR, exist_ok=True)
    # Timestamp in the filename busts the browser cache when a user
    # uploads a new picture - reusing the same filename would otherwise
    # show the old cached image until a hard refresh.
    filename = secure_filename(f"user{session['user_id']}_{int(time.time())}.{ext}")
    file.save(os.path.join(AVATAR_DIR, filename))

    avatar_path = f"/uploads/avatars/{filename}"
    update_avatar(session["user_id"], avatar_path)
    return jsonify({"message": "Avatar updated.", "avatar_path": avatar_path}), 200


@auth_bp.route("/me/avatar", methods=["DELETE"])
@login_required
def delete_avatar():
    old_path = remove_avatar(session["user_id"])

    if old_path:
        # old_path looks like "/uploads/avatars/<filename>" - strip the
        # leading "/uploads/" to find the file inside AVATAR_DIR, same
        # directory upload_avatar() saves into.
        filename = old_path.rsplit("/", 1)[-1]
        file_path = os.path.join(AVATAR_DIR, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError:
                # Not fatal - the DB no longer points at it either way.
                pass

    return jsonify({"message": "Profile picture removed."}), 200

# ── Forgot Password ──────────────────────────────────────────────────────
# Three steps, three routes - none of them require login_required since
# the whole point is to recover access without being logged in.

@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Step 1: user submits their email -> a 6-digit code is generated,
    stored (hashed), and emailed (or printed to the console - see
    auth/email_service.py)."""
    data = request.get_json(silent=True) or {}
    email = data.get("email")

    try:
        request_password_reset(email)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        # Email backend misconfigured/unreachable (e.g. SMTP_USER not
        # set) - log the real reason server-side, but don't leak SMTP
        # internals to the client.
        print(f"[auth] Failed to send password reset email: {e}")
        return jsonify({"error": "Could not send the reset email right now. Please try again later."}), 502
    except Exception as e:
        # Anything else - most commonly a database error (e.g. the
        # password_resets table doesn't exist yet, or is missing a
        # column the code expects). Without this, an exception here
        # would escape uncaught and Flask would return its own HTML
        # error page instead of JSON, which is exactly the
        # "Unexpected token '<'" error this guards against.
        print(f"[auth] Unexpected error in forgot_password: {e}")
        return jsonify({"error": "Something went wrong. Please try again later."}), 500

    # Always the same message whether or not the email was actually
    # found - see request_password_reset()'s docstring for why.
    return jsonify({
        "message": "If that email is registered, a verification code has been sent."
    }), 200


@auth_bp.route("/verify-reset-code", methods=["POST"])
def verify_reset_code_route():
    """Step 4: user submits the 6-digit code -> validated against the
    most recent unused, unexpired code for that email."""
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    code  = data.get("code")

    try:
        reset_id = verify_reset_code(email, code)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[auth] Unexpected error in verify_reset_code_route: {e}")
        return jsonify({"error": "Something went wrong. Please try again later."}), 500

    # reset_id is handed back to the frontend and passed through
    # untouched on the final "set new password" step, so that exact
    # password_resets row (and no other) is the one consumed.
    return jsonify({"message": "Code verified.", "reset_id": reset_id}), 200


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password_route():
    """Step 5: user submits a new password -> updates users.password_hash
    and marks the password_resets row as used so the code can't be
    replayed."""
    data = request.get_json(silent=True) or {}
    reset_id     = data.get("reset_id")
    email        = data.get("email")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if new_password != confirm_password:
        return jsonify({"error": "Passwords do not match."}), 400

    try:
        reset_password(reset_id, email, new_password)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[auth] Unexpected error in reset_password_route: {e}")
        return jsonify({"error": "Something went wrong. Please try again later."}), 500

    return jsonify({"message": "Password reset successfully. Please sign in."}), 200