"""
auth/auth_service.py - Business logic for user registration and login.

This module is the only place that talks directly to the `users` table.
auth_routes.py calls into these functions; nothing here knows about
Flask, sessions, or HTTP requests - that separation keeps this testable
on its own and keeps auth_routes.py thin.
"""

from typing import Optional

from werkzeug.security import generate_password_hash, check_password_hash
from mysql.connector import IntegrityError

from database.db import get_connection


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def create_user(fullname: str, email: str, username: str, password: str) -> dict:
    """
    Insert a new user with a hashed password.
    Raises ValueError (safe to show to the user) if a field is missing,
    the password is too short, or the email/username is already taken.
    """
    fullname = (fullname or "").strip()
    email    = (email or "").strip().lower()
    username = (username or "").strip().lower()

    if not fullname or not email or not username or not password:
        raise ValueError("All fields are required.")
    if "@" not in email:
        raise ValueError("Please enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    password_hash = hash_password(password)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO users (fullname, email, username, password_hash)
                VALUES (%s, %s, %s, %s)
                """,
                (fullname, email, username, password_hash),
            )
            conn.commit()
            user_id = cursor.lastrowid
        except IntegrityError:
            raise ValueError("That email or username is already registered.")
        finally:
            cursor.close()
    finally:
        conn.close()

    return {"id": user_id, "fullname": fullname, "email": email, "username": username}


def find_user_by_login(identifier: str) -> Optional[dict]:
    """Look up a user by username OR email - whichever was typed at login."""
    identifier = (identifier or "").strip().lower()
    if not identifier:
        return None

    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE username = %s OR email = %s LIMIT 1",
            (identifier, identifier),
        )
        user = cursor.fetchone()
        cursor.close()
        return user
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, fullname, email, username, created_at,
                   avatar_path, last_active
            FROM users WHERE id = %s
            """,
            (user_id,),
        )
        user = cursor.fetchone()
        cursor.close()
        return user
    finally:
        conn.close()


def start_session(user_id: int) -> int:
    """
    Insert a new sessions row when a user logs in.
    Returns the new row's id, which the caller stores in the Flask
    session so end_session() knows which row to close out later.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (user_id) VALUES (%s)",
            (user_id,),
        )
        conn.commit()
        session_id = cursor.lastrowid
        cursor.close()
        return session_id
    finally:
        conn.close()


def end_session(session_id: Optional[int]) -> None:
    """
    Mark a sessions row as finished by setting logout_time. MySQL's
    generated `duration` column fills itself in automatically once
    logout_time is set - nothing else to compute here.
    """
    if not session_id:
        return
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET logout_time = NOW() WHERE id = %s",
            (session_id,),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def touch_last_active(user_id: int) -> None:
    """
    Stamp users.last_active to right now. Called on login, and again
    whenever the camera pipeline actually starts for this user (see
    server.py's on_connect), so it reflects real usage rather than just
    "logged in at some point."
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_active = NOW() WHERE id = %s",
            (user_id,),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def update_profile(user_id: int, fullname: str, username: str) -> dict:
    """
    Update a user's fullname/username (the Profile page's "Edit Profile"
    button). Email is intentionally not editable here - it's kept as
    the account's permanent, read-only identifier on the Profile page,
    not an oversight.
    Raises ValueError (safe to show to the user) on bad input or if the
    new username is already taken by someone else.
    """
    fullname = (fullname or "").strip()
    username = (username or "").strip().lower()

    if not fullname or not username:
        raise ValueError("Full name and username are required.")
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters long.")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            # Exclude the current user's own row so re-saving their
            # existing username doesn't trip the "already taken" check.
            cursor.execute(
                "SELECT id FROM users WHERE username = %s AND id != %s",
                (username, user_id),
            )
            if cursor.fetchone():
                raise ValueError("That username is already taken.")

            cursor.execute(
                "UPDATE users SET fullname = %s, username = %s WHERE id = %s",
                (fullname, username, user_id),
            )
            conn.commit()
        except IntegrityError:
            raise ValueError("That username is already taken.")
        finally:
            cursor.close()
    finally:
        conn.close()

    return get_user_by_id(user_id)


def update_password(user_id: int, current_password: str, new_password: str) -> None:
    """
    Change a user's password from the Profile page's "Change Password"
    button. Requires the current password to match first - same
    verify_password() used at login - so a hijacked, already-logged-in
    session can't be used to silently lock the real owner out.
    Raises ValueError (safe to show to the user) on bad input.
    """
    if len(new_password or "") < 8:
        raise ValueError("New password must be at least 8 characters long.")

    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        if not row or not verify_password(current_password or "", row["password_hash"]):
            cursor.close()
            raise ValueError("Current password is incorrect.")

        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(new_password), user_id),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def update_avatar(user_id: int, avatar_path: str) -> None:
    """
    Save the path of a newly uploaded profile picture. The file itself
    is written to disk by auth_routes.py's upload_avatar() - this just
    records where it ended up.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET avatar_path = %s WHERE id = %s",
            (avatar_path, user_id),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def remove_avatar(user_id: int) -> Optional[str]:
    """
    Clear a user's avatar_path so the dashboard falls back to the
    default placeholder avatar. Returns the previous avatar_path (or
    None if there wasn't one) so the caller can delete the file on
    disk too, same pattern as update_avatar() but for the "Remove
    picture" button on the Profile page.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT avatar_path FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        old_path = row["avatar_path"] if row else None

        cursor.execute(
            "UPDATE users SET avatar_path = NULL WHERE id = %s",
            (user_id,),
        )
        conn.commit()
        cursor.close()
        return old_path
    finally:
        conn.close()