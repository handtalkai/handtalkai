"""
feedback/feedback_routes.py - Flask Blueprint exposing the Customer
Support "Feedback" section as a small JSON API:
    POST /api/support/report - save an issue report
    POST /api/feedback       - save free-text feedback and/or a
                                gesture-accuracy star rating

Persists to JSON files (see feedback_service.py) instead of MySQL,
per the prototype's storage requirements for this feature only.
"""

from flask import Blueprint, request, jsonify

from feedback.feedback_service import save_support_report, save_feedback

feedback_bp = Blueprint("feedback", __name__, url_prefix="/api")


@feedback_bp.route("/support/report", methods=["POST"])
def report_issue():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    if not title or not description:
        return jsonify({"error": "Title and description are required."}), 400
    save_support_report(title, description)
    return jsonify({"message": "Issue reported. Thank you!"}), 200


@feedback_bp.route("/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip() or None
    rating = data.get("rating")

    if rating is not None:
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            return jsonify({"error": "Rating must be a number between 1 and 5."}), 400
        if rating < 1 or rating > 5:
            return jsonify({"error": "Rating must be between 1 and 5."}), 400

    if not message and rating is None:
        return jsonify({"error": "Feedback message or rating is required."}), 400

    save_feedback(message=message, rating=rating)
    return jsonify({"message": "Thank you for your feedback!"}), 200
