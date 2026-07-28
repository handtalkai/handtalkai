"""
auth/password_reset_service.py - Business logic for the Forgot Password
flow (request code -> verify code -> set new password).

Kept as its own module, separate from auth_service.py, so none of the
existing register/login/profile functions in that file need to change
at all - this only adds new functions and only touches the new
password_resets table (plus a final UPDATE on users.password_hash at
the very last step, using the same hash_password() auth_service.py
already uses everywhere else).
"""

import random
from datetime import datetime, timedelta
from typing import Optional

from werkzeug.security import generate_password_hash, check_password_hash

from database.db import get_connection
from auth.auth_service import hash_password
from auth.email_service import send_reset_code_email

CODE_LENGTH = 6
CODE_EXPIRY_MINUTES = 10


def _generate_code() -> str:
    """A random 6-digit code, e.g. '482931'. Zero-padded so it's
    always exactly 6 digits even if the number itself is small."""
    return f"{random.randint(0, 999999):06d}"


def _hash_code(code: str) -> str:
    """Reuse Werkzeug's password hasher for the reset code too, so
    a leaked password_resets table can't be used to replay anyone's
    code - same reasoning as hashing users.password_hash."""
    return generate_password_hash(code)


def _verify_code_hash(code: str, code_hash: str) -> bool:
    return check_password_hash(code_hash, code)


def request_password_reset(email: str) -> None:
    """
    Step 1: look up the user by email, generate a 6-digit code, store
    its hash with a 10-minute expiry, and send it.

    Deliberately does NOT raise an error for an unknown email - the
    route always returns the same generic success message either way,
    so this endpoint can't be used to discover which emails are
    registered (a common account-enumeration concern). It simply does
    nothing if the email isn't found.
    """
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("Please enter a valid email address.")

    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE email = %s LIMIT 1", (email,))
        user = cursor.fetchone()

        if not user:
            # Same "do nothing, but don't error" as above - caller
            # still shows the generic "if that email exists..." message.
            cursor.close()
            return

        code = _generate_code()
        code_hash = _hash_code(code)
        expires_at = datetime.now() + timedelta(minutes=CODE_EXPIRY_MINUTES)

        cursor.execute(
            """
            INSERT INTO password_resets (user_id, email, reset_code, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            (user["id"], email, code_hash, expires_at),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()

    # Sent after the DB row is committed, so if email sending fails
    # (e.g. SMTP misconfigured) the code still exists and a retry of
    # this same function will simply issue a fresh one - no orphaned
    # state either way.
    send_reset_code_email(email, code)


def verify_reset_code(email: str, code: str) -> int:
    """
    Step 4: check a submitted code against the most recent unused,
    unexpired reset_code row for this email.
    Returns the password_resets row's id (the caller passes this
    through to reset_password() so that exact row - and no other - is
    the one that gets marked used).
    Raises ValueError (safe to show to the user) if the code is wrong,
    expired, or already used.
    """
    email = (email or "").strip().lower()
    code = (code or "").strip()
    if not email or not code:
        raise ValueError("Please enter the verification code.")

    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, reset_code, expires_at, used_at
            FROM password_resets
            WHERE email = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email,),
        )
        row = cursor.fetchone()
        cursor.close()
    finally:
        conn.close()

    # One deliberately generic error message for every failure mode
    # below (wrong code, expired, already used, no request at all) -
    # being specific here would help an attacker narrow down whether
    # a guess was "close" to correct.
    generic_error = "That code is invalid or has expired. Please request a new one."

    if not row:
        raise ValueError(generic_error)
    if row["used_at"] is not None:
        raise ValueError(generic_error)
    if row["expires_at"] < datetime.now():
        raise ValueError(generic_error)
    if not _verify_code_hash(code, row["reset_code"]):
        raise ValueError(generic_error)

    return row["id"]


def reset_password(reset_id: int, email: str, new_password: str) -> None:
    """
    Step 5: set a new password, then mark the password_resets row as
    used so the same code can never be replayed - even if a user
    re-submits it again before it would otherwise expire.
    Raises ValueError (safe to show to the user) on bad input or if
    reset_id doesn't belong to this email (defensive - shouldn't
    normally happen since the frontend always passes through the id
    verify_reset_code() just returned).
    """
    if len(new_password or "") < 8:
        raise ValueError("New password must be at least 8 characters long.")

    email = (email or "").strip().lower()

    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, user_id, used_at, expires_at
            FROM password_resets
            WHERE id = %s AND email = %s
            LIMIT 1
            """,
            (reset_id, email),
        )
        row = cursor.fetchone()

        if not row or row["used_at"] is not None or row["expires_at"] < datetime.now():
            cursor.close()
            raise ValueError("That reset request is no longer valid. Please start over.")

        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(new_password), row["user_id"]),
        )
        cursor.execute(
            "UPDATE password_resets SET used_at = NOW() WHERE id = %s",
            (reset_id,),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()