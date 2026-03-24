"""
SQLite persistence layer.

One file per user. Zero external dependencies (sqlite3 is in stdlib).
Handles: memories, profile, session metadata.

Schema:
    memories   — every message ever stored
    profile    — single-row JSON blob, updated per session
    sessions   — session start/end timestamps, message count
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    session_id  INTEGER NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS profile (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    data    TEXT    NOT NULL DEFAULT '{}',
    updated TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  INTEGER PRIMARY KEY,
    started_at  TEXT,
    ended_at    TEXT,
    msg_count   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
"""


class MemoryDB:
    """Thin SQLite wrapper for memx persistence."""

    def __init__(self, user_id: str, db_dir: str = "~/.memx"):
        db_dir = os.path.expanduser(db_dir)
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        db_path = os.path.join(db_dir, f"{user_id}.db")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        # Ensure the single profile row exists
        self.conn.execute(
            "INSERT OR IGNORE INTO profile (id, data) VALUES (1, '{}')"
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def insert_memory(
        self,
        text: str,
        timestamp: str,
        role: str,
        session_id: int,
    ) -> int:
        """Insert a memory and return its rowid."""
        cur = self.conn.execute(
            "INSERT INTO memories (text, timestamp, role, session_id) "
            "VALUES (?, ?, ?, ?)",
            (text, timestamp, role, session_id),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_all_memories(self) -> list[dict]:
        """Return all memories as a list of dicts, ordered by id."""
        rows = self.conn.execute(
            "SELECT id, text, timestamp, role, session_id FROM memories ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "timestamp": r["timestamp"],
                "role": r["role"],
                "session_id": r["session_id"],
            }
            for r in rows
        ]

    def memory_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_profile(self) -> dict:
        row = self.conn.execute("SELECT data FROM profile WHERE id = 1").fetchone()
        if row is None:
            return {}
        try:
            return json.loads(row["data"])
        except json.JSONDecodeError:
            return {}

    def set_profile(self, data: dict) -> None:
        self.conn.execute(
            "UPDATE profile SET data = ?, updated = datetime('now') WHERE id = 1",
            (json.dumps(data, ensure_ascii=False),),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: int,
        started_at: str | None = None,
        ended_at: str | None = None,
        msg_count: int | None = None,
    ) -> None:
        existing = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO sessions (session_id, started_at, ended_at, msg_count) "
                "VALUES (?, ?, ?, ?)",
                (session_id, started_at, ended_at, msg_count or 0),
            )
        else:
            parts, vals = [], []
            if started_at is not None:
                parts.append("started_at = ?")
                vals.append(started_at)
            if ended_at is not None:
                parts.append("ended_at = ?")
                vals.append(ended_at)
            if msg_count is not None:
                parts.append("msg_count = ?")
                vals.append(msg_count)
            if parts:
                vals.append(session_id)
                self.conn.execute(
                    f"UPDATE sessions SET {', '.join(parts)} WHERE session_id = ?",
                    vals,
                )
        self.conn.commit()

    def get_latest_session_id(self) -> int:
        row = self.conn.execute(
            "SELECT MAX(session_id) AS sid FROM sessions"
        ).fetchone()
        return row["sid"] if row and row["sid"] is not None else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()
