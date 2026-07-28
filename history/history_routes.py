"""
history/history_routes.py - Flask Blueprint exposing sentence history as a
small JSON API: GET /api/sentences, DELETE /api/sentences/<id>.

Reuses the login_required decorator from auth.auth_routes so these
endpoints require the same active session as everything else. Does not
touch the AI pipeline, camera, or SocketIO frame-streaming loop.
"""

from flask import Blueprint, jsonify, request, session

from auth.auth_routes import login_required
from history.history_service import get_sentences_for_user, delete_sentence

history_bp = Blueprint("history", __name__, url_prefix="/api")


@history_bp.route("/sentences", methods=["GET"])
@login_required
def list_sentences():
    search = request.args.get("search", "").strip() or None
    rows = get_sentences_for_user(session["user_id"], search=search)
    return jsonify({"sentences": rows}), 200


@history_bp.route("/sentences/<int:sentence_id>", methods=["DELETE"])
@login_required
def remove_sentence(sentence_id):
    deleted = delete_sentence(session["user_id"], sentence_id)
    if not deleted:
        return jsonify({"error": "Sentence not found."}), 404
    return jsonify({"message": "Deleted."}), 200