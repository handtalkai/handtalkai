"""
feedback/feedback_service.py - JSON-file persistence for the Customer
Support "Feedback" section (issue reports, free-text feedback, and
gesture accuracy star ratings).

This is the one feature in HandTalk AI that intentionally stores data
in flat JSON files under data/ instead of MySQL, per the prototype's
storage requirements for this feature only.
"""

import json
import os
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")
SUPPORT_REPORTS_FILE = os.path.join(DATA_DIR, "support_reports.json")


def _read_json_list(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _append_json_entry(path, entry):
    os.makedirs(DATA_DIR, exist_ok=True)
    entries = _read_json_list(path)
    entries.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entry


def save_support_report(title, description):
    entry = {
        "type": "issue",
        "title": title,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return _append_json_entry(SUPPORT_REPORTS_FILE, entry)


def save_feedback(message=None, rating=None):
    if message:
        entry = {"type": "feedback", "message": message}
        if rating is not None:
            entry["rating"] = rating
    else:
        entry = {"type": "rating", "rating": rating}
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    return _append_json_entry(FEEDBACK_FILE, entry)
