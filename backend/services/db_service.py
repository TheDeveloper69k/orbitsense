"""
db_service.py

Small SQLite wrapper for the activity_log table. No ORM — plain sqlite3
so it's easy to inspect/debug during the hackathon demo.
"""

import sqlite3
import os
from datetime import datetime, timezone
from contextlib import contextmanager

# DB file lives at backend/db/orbitsense.db
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")
DB_PATH = os.path.join(DB_DIR, "orbitsense.db")


def init_db():
    """Create the db directory + activity_log table if they don't exist yet.
    Safe to call every time the app starts."""
    os.makedirs(DB_DIR, exist_ok=True)
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                step INTEGER,
                object TEXT,
                status TEXT NOT NULL,
                message TEXT
            )
            """
        )
        conn.commit()


@contextmanager
def get_connection():
    """Context-managed connection so callers never forget to close it."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_log(result: dict):
    """
    Insert one pipeline result into activity_log.
    Expected shape: {"step": int, "object": str, "status": str, "message": str}
    Missing keys are tolerated and stored as None.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO activity_log (timestamp, step, object, status, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                result.get("step"),
                result.get("object"),
                result.get("status", "unknown"),
                result.get("message", ""),
            ),
        )
        conn.commit()
        return cur.lastrowid


def fetch_all_logs():
    """Return every row in activity_log, most recent first, as a list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_log ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def fetch_logs_by_status(status: str):
    """Optional helper — e.g. pull only 'violation' rows."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_log WHERE status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
        return [dict(row) for row in rows]