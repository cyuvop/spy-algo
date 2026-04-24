"""
SQLite-backed persistence for daily regime state and run log.

Schema
------
signal_state (sleeve, date PK, state, payload_json)
run_log      (run_timestamp PK, run_date, success, notes)

Design notes
------------
- No global connection; open/close per call (safe for daily runner).
- All functions create tables on first use via _init_db().
- payload is stored as JSON text and decoded back to a plain dict on read.
"""

import json
import sqlite3
from datetime import datetime, timezone


_DDL = """
CREATE TABLE IF NOT EXISTS signal_state (
  sleeve       TEXT NOT NULL,
  date         TEXT NOT NULL,
  state        TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (sleeve, date)
);

CREATE TABLE IF NOT EXISTS run_log (
  run_timestamp TEXT PRIMARY KEY,
  run_date      TEXT NOT NULL,
  success       INTEGER NOT NULL,
  notes         TEXT
);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they do not yet exist."""
    conn.executescript(_DDL)


# ── Public API ────────────────────────────────────────────────────────────────


def get_last_state(sleeve: str, db_path: str = "data/state.db") -> dict | None:
    """Return the most recent signal_state row for *sleeve* as a dict, or None."""
    with sqlite3.connect(db_path) as conn:
        _init_db(conn)
        row = conn.execute(
            """
            SELECT sleeve, date, state, payload_json
            FROM signal_state
            WHERE sleeve = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (sleeve,),
        ).fetchone()

    if row is None:
        return None

    return {
        "sleeve": row[0],
        "date": row[1],
        "state": row[2],
        "payload": json.loads(row[3]),
    }


def save_state(
    sleeve: str,
    date: str,
    state: str,
    payload: dict,
    db_path: str = "data/state.db",
) -> None:
    """Insert or replace a row in signal_state (upsert on sleeve+date PK)."""
    payload_json = json.dumps(payload)
    with sqlite3.connect(db_path) as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO signal_state (sleeve, date, state, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (sleeve, date, state, payload_json),
        )


def has_run_today(date: str, db_path: str = "data/state.db") -> bool:
    """Return True if run_log contains at least one row with run_date == date."""
    with sqlite3.connect(db_path) as conn:
        _init_db(conn)
        count = conn.execute(
            "SELECT COUNT(*) FROM run_log WHERE run_date = ?",
            (date,),
        ).fetchone()[0]
    return count > 0


def mark_run(
    date: str,
    success: bool,
    notes: str = "",
    db_path: str = "data/state.db",
) -> None:
    """Append a row to run_log. run_timestamp is UTC ISO-8601."""
    run_timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT INTO run_log (run_timestamp, run_date, success, notes)
            VALUES (?, ?, ?, ?)
            """,
            (run_timestamp, date, int(success), notes),
        )
