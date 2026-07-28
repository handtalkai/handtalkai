"""
favorites/favorites_routes.py - Flask Blueprint exposing favorites as a
small JSON API:
    GET    /api/favorites               list the current user's favorites
    POST   /api/favorites/<sentence_id> star a sentence
    DELETE /api/favorites/<sentence_id> unstar a sentence

Reuses the login_required decorator from auth.auth_routes, same as
history_routes.py, so these endpoints require the same active session as
everything else. Does not touch the AI pipeline, camera, or SocketIO
frame-streaming loop.
"""

from flask import Blueprint, jsonify, session

from auth.auth_routes import login_required
from favorites.favorites_service import (
    add_favorite,
    remove_favorite,
    get_favorites_for_user,
)

favorites_bp = Blueprint("favorites", __name__, url_prefix="/api")


@favorites_bp.route("/favorites", methods=["GET"])
@login_required
def list_favorites():
    rows = get_favorites_for_user(session["user_id"])
    return jsonify({"favorites": rows}), 200


@favorites_bp.route("/favorites/<int:sentence_id>", methods=["POST"])
@login_required
def favorite_sentence(sentence_id):
    favorite = add_favorite(session["user_id"], sentence_id)
    if favorite is None:
        return jsonify({"error": "Sentence not found."}), 404
    return jsonify({"favorite": favorite}), 201


@favorites_bp.route("/favorites/<int:sentence_id>", methods=["DELETE"])
@login_required
def unfavorite_sentence(sentence_id):
    removed = remove_favorite(session["user_id"], sentence_id)
    if not removed:
        return jsonify({"error": "Favorite not found."}), 404
    return jsonify({"message": "Removed."}), 200