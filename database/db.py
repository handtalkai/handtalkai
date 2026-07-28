"""
database/db.py - MySQL connection helper for HandTalk AI.

This is the single place that knows how to open a connection to MySQL.
The (upcoming) authentication and sentence-history modules will import
get_connection() from here. Nothing in src/ (the AI pipeline) imports
from this module, and this module never imports from src/ - the two
systems stay fully decoupled.
"""

import mysql.connector
from mysql.connector import Error

from database.db_config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


def get_connection():
    """
    Open and return a new MySQL connection.
    Caller is responsible for closing it when done, e.g.:

        conn = get_connection()
        try:
            ...
        finally:
            conn.close()
    """
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
        )
    except Error as e:
        raise RuntimeError(f"Could not connect to MySQL database '{DB_NAME}': {e}")


def test_connection():
    """Manual sanity check: run with `python -m database.db`"""
    conn = get_connection()
    print(f"[database] Connected to MySQL successfully: {conn.is_connected()}")
    conn.close()


if __name__ == "__main__":
    test_connection()