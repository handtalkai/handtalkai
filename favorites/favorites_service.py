"""
favorites/favorites_service.py - Database operations for the `favorites`
table.

This module is the only place that talks directly to the favorites
table. It does join against sentence_history for display purposes (the
sentence text, when it was originally spoken, which session it came
from) but never writes to sentence_history - that table stays owned by
history_service.py.
"""

from typing import Optional, List

from database.db import get_connection


def add_favorite(user_id: int, sentence_id: int) -> Optional[dict]:
    """
    Mark a sentence as a favorite for this user.

    Confirms the sentence actually belongs to this user first, so a
    user can never favorite (or even probe for the existence of)
    another user's sentence by guessing an id.

    Idempotent: favoriting an already-favorited sentence just returns
    successfully instead of erroring, thanks to the unique key on
    (user_id, sentence_id) plus ON DUPLICATE KEY UPDATE as a no-op.

    Returns a small dict on success, or None if the sentence doesn't
    exist / doesn't belong to this user.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM sentence_history WHERE id = %s AND user_id = %s",
            (sentence_id, user_id),
        )
        if cursor.fetchone() is None:
            cursor.close()
            return None

        cursor.execute(
            """
            INSERT INTO favorites (user_id, sentence_id)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE id = id
            """,
            (user_id, sentence_id),
        )
        conn.commit()
        cursor.close()
        return {"user_id": user_id, "sentence_id": sentence_id}
    finally:
        conn.close()


def remove_favorite(user_id: int, sentence_id: int) -> bool:
    """
    Unmark a sentence as a favorite. Scoped to (user_id AND sentence_id)
    so a user can never remove another user's favorite.
    Returns True if a row was actually deleted.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM favorites WHERE user_id = %s AND sentence_id = %s",
            (user_id, sentence_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    finally:
        conn.close()


def get_favorites_for_user(user_id: int) -> List[dict]:
    """
    Return this user's favorited sentences, most recently favorited
    first, joined with sentence_history for display (sentence text,
    when it was originally spoken, which session it came from).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                f.id          AS favorite_id,
                f.created_at  AS favorited_at,
                s.id          AS sentence_id,
                s.sentence    AS sentence,
                s.session_id  AS session_id,
                s.created_at  AS created_at
            FROM favorites f
            JOIN sentence_history s ON s.id = f.sentence_id
            WHERE f.user_id = %s
            ORDER BY f.created_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    # Same datetime -> ISO string treatment as history_service.py, so the
    # frontend can parse both endpoints' timestamps identically.
    for row in rows:
        for col in ("favorited_at", "created_at"):
            if row.get(col) is not None:
                row[col] = row[col].isoformat()
    return rows