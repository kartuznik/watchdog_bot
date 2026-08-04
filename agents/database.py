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
    """Open SQLite with WAL + busy_timeout for safe concurrent bot/worker/admin writes."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roles (
                user_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                granted_by TEXT,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_role ON roles(role)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS async_tasks (
                task_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                payload TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'queued',
                result TEXT DEFAULT '',
                error TEXT DEFAULT '',
                stage TEXT DEFAULT '',
                chat_id TEXT DEFAULT '',
                progress_message_id INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_async_tasks_user_id ON async_tasks(user_id)"
        )
        _ensure_column(conn, "async_tasks", "stage", "TEXT DEFAULT ''")
        _ensure_column(conn, "async_tasks", "chat_id", "TEXT DEFAULT ''")
        _ensure_column(conn, "async_tasks", "progress_message_id", "INTEGER DEFAULT 0")
        conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row[1]) for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


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


def list_users(limit: int = 1000) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 5000))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, created_at, last_seen
            FROM users
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_users() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    return int(row["c"] if row else 0)


def count_conversations() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM conversations").fetchone()
    return int(row["c"] if row else 0)


def count_user_requests() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM conversations WHERE role='user'"
        ).fetchone()
    return int(row["c"] if row else 0)


def count_error_responses() -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM conversations
            WHERE role='assistant' AND content LIKE '%Произошла ошибка%'
            """
        ).fetchone()
    return int(row["c"] if row else 0)


def create_async_task(
    *,
    task_id: str,
    user_id: int,
    task_type: str,
    payload: str,
    chat_id: int | str = "",
    progress_message_id: int = 0,
    stage: str = "queued",
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO async_tasks (
                task_id, user_id, task_type, payload, status, stage, chat_id, progress_message_id, updated_at
            )
            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                task_id,
                str(user_id),
                task_type.strip(),
                payload.strip(),
                stage.strip(),
                str(chat_id),
                int(progress_message_id or 0),
            ),
        )
        conn.commit()


def update_async_task_status(
    task_id: str,
    *,
    status: str,
    result: str = "",
    error: str = "",
    stage: str | None = None,
) -> int:
    with get_connection() as conn:
        if stage is None:
            cur = conn.execute(
                """
                UPDATE async_tasks
                SET status=?, result=?, error=?, updated_at=CURRENT_TIMESTAMP
                WHERE task_id=?
                """,
                (status.strip(), result, error, task_id.strip()),
            )
        else:
            cur = conn.execute(
                """
                UPDATE async_tasks
                SET status=?, result=?, error=?, stage=?, updated_at=CURRENT_TIMESTAMP
                WHERE task_id=?
                """,
                (status.strip(), result, error, stage.strip(), task_id.strip()),
            )
        conn.commit()
    return int(cur.rowcount or 0)


def update_async_task_stage(task_id: str, stage: str) -> int:
    """Update stage only when it actually changes (reduces poller/edit spam)."""
    clean_stage = stage.strip()
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE async_tasks
            SET stage=?, updated_at=CURRENT_TIMESTAMP
            WHERE task_id=? AND IFNULL(stage, '') != ?
            """,
            (clean_stage, task_id.strip(), clean_stage),
        )
        conn.commit()
    return int(cur.rowcount or 0)


def get_async_task(task_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT task_id, user_id, task_type, payload, status, result, error,
                   stage, chat_id, progress_message_id, created_at, updated_at
            FROM async_tasks
            WHERE task_id=?
            """,
            (task_id.strip(),),
        ).fetchone()
    return dict(row) if row else None
