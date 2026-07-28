"""
auth/email_config.py - Email delivery settings for HandTalk AI's
Forgot Password feature.

Kept separate from database/db_config.py on purpose, same reasoning as
that file's own docstring: nothing here can ever affect the database
layer, the AI pipeline, or anything else, and vice versa.

Two backends are supported, switched with EMAIL_BACKEND:

  "console" (default) - no setup required. The verification code is
      printed to the terminal server.py is running in, formatted like
      a real email. This is the right choice while you're developing
      or demoing locally - there's no SMTP account to configure.

  "smtp" - sends a real email via any standard SMTP provider (Gmail,
      Outlook, SendGrid, Mailgun, your school's mail server, etc.).
      Fill in the settings below, or set the matching environment
      variables, then set EMAIL_BACKEND=smtp.

      For Gmail specifically: you cannot use your normal account
      password here - Google requires a 16-character "App Password"
      (Google Account -> Security -> 2-Step Verification -> App
      Passwords). Regular passwords are rejected by Gmail's SMTP.
"""

import os

EMAIL_BACKEND = os.environ.get("EMAIL_BACKEND", "console")  # "console" or "smtp"

SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER     = os.environ.get("SMTP_USER", "")        # <-- your sending email address
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")    # <-- app password, not your login password
SMTP_FROM     = os.environ.get("SMTP_FROM", SMTP_USER or "noreply@handtalk.ai")
SMTP_USE_TLS  = os.environ.get("SMTP_USE_TLS", "1") != "0"