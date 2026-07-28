"""
user_profile/profile_routes.py - Flask Blueprint exposing the Profile
page's statistics as a small JSON API: GET /api/profile/stats.

Reuses the login_required decorator from auth.auth_routes, same as
history_routes.py and favorites_routes.py, so this endpoint requires the
same active session as everything else.
"""

from flask import Blueprint, jsonify, session

from auth.auth_routes import login_required
from user_profile.profile_service import get_profile_stats

profile_bp = Blueprint("profile", __name__, url_prefix="/api")


@profile_bp.route("/profile/stats", methods=["GET"])
@login_required
def profile_stats():
    stats = get_profile_stats(session["user_id"])
    return jsonify({"stats": stats}), 200