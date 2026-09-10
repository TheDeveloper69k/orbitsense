"""
seed_db.py

Creates (or reuses) a SQLite database with an `activity_log` table for
OrbitSense. Each row records one detected pour event: which step it
corresponds to, which object was poured, whether it was ok or a violation,
and the voice/log message shown to the user.

Usage:
    python seed_db.py
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Path to the SQLite DB file (created alongside this script by default)
DB_PATH = Path(__file__).resolve().parent / "orbitsense.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS activity_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    object    TEXT    NOT NULL,
    status    TEXT    NOT NULL CHECK (status IN ('ok', 'violation')),
    message   TEXT    NOT NULL
);
"""

# A couple of example rows so the table isn't empty after seeding.
SAMPLE_ROWS = [
    (
        datetime.now(timezone.utc).isoformat(),
        1,
        "water",
        "ok",
        "Good, water poured first. You may now proceed to pour the acid.",
    ),
    (
        datetime.now(timezone.utc).isoformat(),
        2,
        "acid",
        "ok",
        "Acid poured correctly after water. Step complete.",
    ),
]


def create_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database file."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_table(conn: sqlite3.Connection) -> None:
    """Create the activity_log table if it doesn't already exist."""
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()


def log_activity(
    conn: sqlite3.Connection,
    step: int,
    object_name: str,
    status: str,
    message: str,
    timestamp: str | None = None,
) -> int:
    """
    Insert a single activity log row. Returns the new row's id.

    status must be either 'ok' or 'violation'.
    """
    if status not in ("ok", "violation"):
        raise ValueError("status must be 'ok' or 'violation'")

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    cursor = conn.execute(
        """
        INSERT INTO activity_log (timestamp, step, object, status, message)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ts, step, object_name, status, message),
    )
    conn.commit()
    return cursor.lastrowid


def seed_sample_data(conn: sqlite3.Connection) -> None:
    """Insert a couple of example rows (only if the table is empty)."""
    count = conn.execute("SELECT COUNT(*) FROM activity_log").fetchone()[0]
    if count == 0:
        conn.executemany(
            """
            INSERT INTO activity_log (timestamp, step, object, status, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            SAMPLE_ROWS,
        )
        conn.commit()
        print(f"Seeded {len(SAMPLE_ROWS)} sample row(s) into activity_log.")
    else:
        print(f"activity_log already has {count} row(s); skipping sample seed.")


def main() -> None:
    conn = create_connection()
    try:
        create_table(conn)
        print(f"Database ready at: {DB_PATH}")
        seed_sample_data(conn)

        # Quick sanity check: print out current contents.
        rows = conn.execute(
            "SELECT id, timestamp, step, object, status, message FROM activity_log ORDER BY id"
        ).fetchall()
        print("\nCurrent activity_log contents:")
        for row in rows:
            print(row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()