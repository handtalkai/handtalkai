"""
database/db_config.py - MySQL connection settings for HandTalk AI's login
system and history storage.

This is intentionally separate from src/config.py, which belongs to the
AI pipeline (model paths, thresholds, camera settings, etc.). Keeping
them apart means nothing here can ever affect the gesture recognition
system, and vice versa.

Fill in your local MySQL credentials below, or set environment variables
of the same name (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME) if you'd rather
not hardcode them - useful once you deploy this somewhere other than your
own machine.
"""

import os

DB_HOST     = os.environ.get("DB_HOST", "localhost")
DB_PORT     = int(os.environ.get("DB_PORT", 3306))
DB_USER     = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")  # <-- put your MySQL password here
DB_NAME     = os.environ.get("DB_NAME", "handtalk_ai")