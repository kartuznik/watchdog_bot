"""SQLite helpers adapted from telegram-ai-bot database module."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.db"


def _resolve_db_path() -> Path:
    raw = os.getenv("AGENT_DB_PATH", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


DB_PATH = _resolve_db_path()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_anchors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                context_snippet TEXT NOT NULL,
                message_ref INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_anchors_user_id ON conversation_anchors(user_id)"
        )
        conn.commit()


def ensure_user(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id)
            VALUES (?)
            ON CONFLICT(user_id) DO UPDATE SET last_seen=CURRENT_TIMESTAMP
            """,
            (str(user_id),),
        )
        conn.commit()


def list_recent_conversations(limit: int = 100, user_id: int | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    with get_connection() as conn:
        if user_id is None:
            rows = conn.execute(
                """
                SELECT id, user_id, role, content, created_at
                FROM conversations
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, role, content, created_at
                FROM conversations
                WHERE user_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
    return [dict(r) for r in rows]


def clear_conversations(user_id: int | None = None) -> int:
    with get_connection() as conn:
        if user_id is None:
            cur = conn.execute("DELETE FROM conversations")
        else:
            cur = conn.execute("DELETE FROM conversations WHERE user_id=?", (str(user_id),))
        conn.commit()
        return int(cur.rowcount or 0)


def list_anchors(limit: int = 100, user_id: int | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    with get_connection() as conn:
        if user_id is None:
            rows = conn.execute(
                """
                SELECT id, user_id, title, context_snippet, message_ref, created_at
                FROM conversation_anchors
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, title, context_snippet, message_ref, created_at
                FROM conversation_anchors
                WHERE user_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
    return [dict(r) for r in rows]
