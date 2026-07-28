"""
auth/email_service.py - Sends the Forgot Password verification code.

Two backends, picked by auth.email_config.EMAIL_BACKEND:
    "console" (default) - prints the email to the terminal instead of
        actually sending it. Zero setup, perfect for local dev/demo.
    "smtp" - sends a real email via smtplib using the settings in
        auth/email_config.py.

auth_routes.py / password_reset_service.py only ever call
send_reset_code_email() - they don't know or care which backend is
active. Swapping backends is a one-line env var change
(EMAIL_BACKEND=smtp), not a code change.
"""

import smtplib
from email.message import EmailMessage

from auth.email_config import (
    EMAIL_BACKEND, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    SMTP_FROM, SMTP_USE_TLS,
)

SUBJECT = "HandTalk AI Password Reset"


def _build_body(code: str) -> str:
    return (
        "Your verification code is:\n\n"
        f"{code}\n\n"
        "This code will expire in 10 minutes.\n\n"
        "If you did not request this password reset, please ignore this email."
    )


def _send_console(to_email: str, code: str) -> None:
    """Default backend: print the email to the server console instead
    of actually sending it. Lets you test the whole Forgot Password
    flow end-to-end with zero email setup - just read the code from
    the terminal server.py is running in."""
    border = "=" * 60
    print(border)
    print("[email_service] EMAIL_BACKEND=console - no real email sent.")
    print(f"To:      {to_email}")
    print(f"Subject: {SUBJECT}")
    print("-" * 60)
    print(_build_body(code))
    print(border)


def _send_smtp(to_email: str, code: str) -> None:
    """Real backend: sends an actual email over SMTP. Requires
    SMTP_USER/SMTP_PASSWORD (and usually an app password, not your
    normal login password - see auth/email_config.py) to be set."""
    if not SMTP_USER or not SMTP_PASSWORD:
        raise RuntimeError(
            "EMAIL_BACKEND=smtp but SMTP_USER/SMTP_PASSWORD are not set. "
            "Fill them in auth/email_config.py or as environment variables."
        )

    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(_build_body(code))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
        if SMTP_USE_TLS:
            server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


def send_reset_code_email(to_email: str, code: str) -> None:
    """
    Send (or print) the password-reset verification code.
    Raises RuntimeError if EMAIL_BACKEND=smtp and sending fails -
    callers should catch this and show a generic "couldn't send the
    email right now" message rather than leaking SMTP internals.
    """
    if EMAIL_BACKEND == "smtp":
        _send_smtp(to_email, code)
    else:
        _send_console(to_email, code)