"""
user_profile/profile_service.py - Read-only statistics for the Profile
page's "Statistics" panel.

Named user_profile/ (not profile/) deliberately - server.py puts the
project root on sys.path, and profile is the name of a Python stdlib
module (the performance profiler), so a top-level profile/ package
here would shadow it.

This module owns no table of its own. It only reads across sessions,
sentence_history, and favorites - each still owned by auth_service.py,
history_service.py, and favorites_service.py respectively - the same
read-only-join pattern favorites_service.py already uses against
sentence_history.
"""

from collections import Counter
from typing import Optional

from database.db import get_connection


def get_profile_stats(user_id: int) -> dict:
    """
    Return the five numbers shown on the Profile page:
    total_sessions, total_sentences, total_favorites,
    avg_duration_seconds, and most_used_letter.

    most_used_letter is computed by tallying letter frequency across
    all of this user's saved sentences (spaces ignored). There's no
    per-gesture log table yet, so this reflects letters in *spoken*
    sentences rather than raw camera detections - a reasonable stand-in
    without adding a new table to the AI pipeline's hot path.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT COUNT(*) AS n FROM sessions WHERE user_id = %s",
            (user_id,),
        )
        total_sessions = cursor.fetchone()["n"]

        cursor.execute(
            "SELECT COUNT(*) AS n FROM sentence_history WHERE user_id = %s",
            (user_id,),
        )
        total_sentences = cursor.fetchone()["n"]

        cursor.execute(
            "SELECT COUNT(*) AS n FROM favorites WHERE user_id = %s",
            (user_id,),
        )
        total_favorites = cursor.fetchone()["n"]

        cursor.execute(
            """
            SELECT AVG(duration) AS avg_duration
            FROM sessions
            WHERE user_id = %s AND duration IS NOT NULL
            """,
            (user_id,),
        )
        avg_duration = cursor.fetchone()["avg_duration"]

        cursor.execute(
            "SELECT sentence FROM sentence_history WHERE user_id = %s",
            (user_id,),
        )
        sentences = [row["sentence"] for row in cursor.fetchall()]

        cursor.close()
    finally:
        conn.close()

    return {
        "total_sessions":       total_sessions,
        "total_sentences":      total_sentences,
        "total_favorites":      total_favorites,
        "avg_duration_seconds": float(avg_duration) if avg_duration is not None else None,
        "most_used_letter":     _most_used_letter(sentences),
    }


def _most_used_letter(sentences) -> Optional[str]:
    counts = Counter()
    for sentence in sentences:
        for ch in sentence.upper():
            if ch.isalpha():
                counts[ch] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]