"""
settings/settings_service.py - Business logic for the Settings page's
AI & Speech, Camera, and Dashboard Preference toggles.

This module is the only place that talks directly to the user_settings
table, same separation auth_service.py and profile_service.py already
use against users/sessions/sentence_history. settings_routes.py calls
into these functions; nothing here knows about Flask, sessions, or HTTP
requests.

Each user gets exactly one row (enforced by the UNIQUE key on user_id
in schema.sql). get_settings() creates that row with defaults on first
access, so callers never have to special-case "no settings yet."
"""

from typing import Optional

from database.db import get_connection

# Keep these in sync with schema.sql's DEFAULT values - used when a
# user's row doesn't exist yet (get_settings' first-time insert) and
# by reset_settings() ("Reset to Default" button).
DEFAULTS = {
    "auto_speak":         False,
    "speech_speed":       1.0,
    "speech_volume":      100,
    "camera_size":        "medium",
    "show_analytics":     True,
    "show_history_panel": True,
}

_ALLOWED_CAMERA_SIZES = {"small", "medium", "large"}

# Columns a caller is allowed to update via update_settings(). Listed
# explicitly (rather than trusting whatever keys arrive from the
# request) so a malformed request body can't be used to write to an
# arbitrary column.
_EDITABLE_FIELDS = (
    "auto_speak", "speech_speed", "speech_volume", "camera_size",
    "show_analytics", "show_history_panel",
)


def _row_to_dict(row: dict) -> dict:
    """Normalize a DB row into the JSON-friendly shape the frontend expects."""
    return {
        "auto_speak":         bool(row["auto_speak"]),
        "speech_speed":       float(row["speech_speed"]),
        "speech_volume":      int(row["speech_volume"]),
        "camera_size":        row["camera_size"],
        "show_analytics":     bool(row["show_analytics"]),
        "show_history_panel": bool(row["show_history_panel"]),
    }


def get_settings(user_id: int) -> dict:
    """
    Return this user's settings, creating a default row first if one
    doesn't exist yet (e.g. a user who registered before the Settings
    page existed, or any user's very first visit to it).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM user_settings WHERE user_id = %s", (user_id,)
        )
        row = cursor.fetchone()

        if row is None:
            cursor.execute(
                """
                INSERT INTO user_settings
                    (user_id, auto_speak, speech_speed, speech_volume,
                     camera_size, show_analytics, show_history_panel)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    DEFAULTS["auto_speak"],
                    DEFAULTS["speech_speed"],
                    DEFAULTS["speech_volume"],
                    DEFAULTS["camera_size"],
                    DEFAULTS["show_analytics"],
                    DEFAULTS["show_history_panel"],
                ),
            )
            conn.commit()
            cursor.execute(
                "SELECT * FROM user_settings WHERE user_id = %s", (user_id,)
            )
            row = cursor.fetchone()

        cursor.close()
    finally:
        conn.close()

    return _row_to_dict(row)


def update_settings(user_id: int, **fields) -> dict:
    """
    Update one or more of this user's settings (the Settings page's
    "Save Settings" button). Unknown keys are ignored rather than
    raising, so the frontend can send the whole settings object and
    not worry about extra/legacy keys.

    Raises ValueError (safe to show to the user) on invalid input.
    """
    updates = {k: v for k, v in fields.items() if k in _EDITABLE_FIELDS and v is not None}
    if not updates:
        return get_settings(user_id)

    if "speech_speed" in updates:
        try:
            speed = float(updates["speech_speed"])
        except (TypeError, ValueError):
            raise ValueError("Speech speed must be a number.")
        if not (0.5 <= speed <= 2.0):
            raise ValueError("Speech speed must be between 0.5x and 2.0x.")
        updates["speech_speed"] = speed

    if "speech_volume" in updates:
        try:
            volume = int(updates["speech_volume"])
        except (TypeError, ValueError):
            raise ValueError("Speech volume must be a number.")
        if not (0 <= volume <= 100):
            raise ValueError("Speech volume must be between 0 and 100.")
        updates["speech_volume"] = volume

    if "camera_size" in updates:
        if updates["camera_size"] not in _ALLOWED_CAMERA_SIZES:
            raise ValueError("Camera size must be small, medium, or large.")

    for bool_field in ("auto_speak", "show_analytics", "show_history_panel"):
        if bool_field in updates:
            updates[bool_field] = bool(updates[bool_field])

    # Make sure the row exists before updating it (same first-visit
    # case get_settings() handles).
    get_settings(user_id)

    set_clause = ", ".join(f"{col} = %s" for col in updates)
    params = list(updates.values()) + [user_id]

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE user_settings SET {set_clause} WHERE user_id = %s",
            params,
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()

    return get_settings(user_id)


def reset_settings(user_id: int) -> dict:
    """Reset this user's settings back to the defaults above (the
    Settings page's "Reset to Default" button)."""
    return update_settings(user_id, **DEFAULTS)
