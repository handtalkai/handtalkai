"""
history/history_service.py - Database operations for sentence_history.

This module is the only place that talks directly to the
sentence_history table. history_routes.py calls into these functions;
server.py also calls save_sentence() directly from the AI pipeline's
speak_text() method, the moment a sentence is actually spoken.

get_sentences_for_user() also LEFT JOINs the favorites table so each row
comes back with an is_favorited flag for the History page's star icon.
That table is still owned by favorites_service.py - this is read-only.
"""

from typing import Optional, List

from database.db import get_connection


def save_sentence(user_id: int, sentence: str, session_id: Optional[int] = None) -> Optional[int]:
    """
    Insert a finalized/spoken sentence into sentence_history.
    Returns the new row's id, or None if there was nothing to save.
    """
    sentence = (sentence or "").strip()
    if not sentence or not user_id:
        return None

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sentence_history (user_id, session_id, sentence)
            VALUES (%s, %s, %s)
            """,
            (user_id, session_id, sentence),
        )
        conn.commit()
        row_id = cursor.lastrowid
        cursor.close()
        return row_id
    finally:
        conn.close()


def get_sentences_for_user(user_id: int, search: Optional[str] = None) -> List[dict]:
    """
    Return this user's sentence history, most recent first.
    If `search` is given, only rows whose sentence contains it (case-
    insensitive substring match) are returned.
    Each row includes is_favorited (1/0) so the History page can render
    the star icon in the right state without a second request.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        if search:
            cursor.execute(
                """
                SELECT
                    s.id, s.session_id, s.sentence, s.created_at,
                    (f.id IS NOT NULL) AS is_favorited
                FROM sentence_history s
                LEFT JOIN favorites f
                    ON f.sentence_id = s.id AND f.user_id = s.user_id
                WHERE s.user_id = %s AND s.sentence LIKE %s
                ORDER BY s.created_at DESC
                """,
                (user_id, f"%{search}%"),
            )
        else:
            cursor.execute(
                """
                SELECT
                    s.id, s.session_id, s.sentence, s.created_at,
                    (f.id IS NOT NULL) AS is_favorited
                FROM sentence_history s
                LEFT JOIN favorites f
                    ON f.sentence_id = s.id AND f.user_id = s.user_id
                WHERE s.user_id = %s
                ORDER BY s.created_at DESC
                """,
                (user_id,),
            )
        rows = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    # created_at comes back as a datetime object - convert to a plain
    # ISO string so it serializes predictably in the JSON response.
    for row in rows:
        if row.get("created_at") is not None:
            row["created_at"] = row["created_at"].isoformat()
    return rows


def delete_sentence(user_id: int, sentence_id: int) -> bool:
    """
    Delete one sentence row. Scoped to (id AND user_id) so a user can
    never delete another user's row, even by guessing an id.
    Returns True if a row was actually deleted.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM sentence_history WHERE id = %s AND user_id = %s",
            (sentence_id, user_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    finally:
        conn.close()