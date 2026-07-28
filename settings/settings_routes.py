"""
settings/settings_routes.py - Flask Blueprint exposing the Settings
page as a small JSON API:
    GET   /api/settings        - load this user's current settings
    PUT   /api/settings        - save changes from the Settings page
    POST  /api/settings/reset  - reset to defaults

Reuses the login_required decorator from auth.auth_routes, same as
profile_routes.py, history_routes.py, and favorites_routes.py, so these
endpoints require the same active session as everything else.
"""

from flask import Blueprint, request, jsonify, session

from auth.auth_routes import login_required
from config import ENABLE_LOGIN
from settings.settings_service import DEFAULTS, get_settings, update_settings, reset_settings

settings_bp = Blueprint("settings", __name__, url_prefix="/api")


@settings_bp.route("/settings", methods=["GET"])
@login_required
def fetch_settings():
    if not ENABLE_LOGIN or session["user_id"] is None:
        return jsonify({"settings": DEFAULTS}), 200
    settings = get_settings(session["user_id"])
    return jsonify({"settings": settings}), 200


@settings_bp.route("/settings", methods=["PUT"])
@login_required
def save_settings():
    if not ENABLE_LOGIN or session["user_id"] is None:
        return jsonify({"message": "Settings saved.", "settings": DEFAULTS}), 200
    data = request.get_json(silent=True) or {}
    try:
        settings = update_settings(
            user_id=session["user_id"],
            auto_speak=data.get("auto_speak"),
            speech_speed=data.get("speech_speed"),
            speech_volume=data.get("speech_volume"),
            camera_size=data.get("camera_size"),
            show_analytics=data.get("show_analytics"),
            show_history_panel=data.get("show_history_panel"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "Settings saved.", "settings": settings}), 200


@settings_bp.route("/settings/reset", methods=["POST"])
@login_required
def reset():
    if not ENABLE_LOGIN or session["user_id"] is None:
        return jsonify({"message": "Settings reset to default.", "settings": DEFAULTS}), 200
    settings = reset_settings(session["user_id"])
    return jsonify({"message": "Settings reset to default.", "settings": settings}), 200
