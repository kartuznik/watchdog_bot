"""SQLite helpers adapted from telegram-ai-bot database module."""

from __future__ import annotations

import csv
import io
import json
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


def get_retention_days() -> int:
    raw = os.getenv("DATA_RETENTION_DAYS", "90").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 90


def soft_delete_gate_enabled() -> bool:
    """When true, soft-deleted users are blocked from the bot (off by default)."""
    return os.getenv("SOFT_DELETE_GATE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                router_decision TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_user_id ON usage_events(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_created_at ON usage_events(created_at)"
        )
        _ensure_column(conn, "async_tasks", "stage", "TEXT DEFAULT ''")
        _ensure_column(conn, "async_tasks", "chat_id", "TEXT DEFAULT ''")
        _ensure_column(conn, "async_tasks", "progress_message_id", "INTEGER DEFAULT 0")
        _ensure_column(conn, "users", "deleted_at", "TIMESTAMP")
        _ensure_column(conn, "conversations", "deleted_at", "TIMESTAMP")
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


def is_user_soft_deleted(user_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT deleted_at FROM users WHERE user_id=?",
            (str(user_id),),
        ).fetchone()
    if not row:
        return False
    return bool(row["deleted_at"])


def list_recent_conversations(
    limit: int = 100,
    user_id: int | None = None,
    *,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    deleted_clause = "" if include_deleted else "AND deleted_at IS NULL"
    with get_connection() as conn:
        if user_id is None:
            rows = conn.execute(
                f"""
                SELECT id, user_id, role, content, created_at, deleted_at
                FROM conversations
                WHERE 1=1 {deleted_clause}
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT id, user_id, role, content, created_at, deleted_at
                FROM conversations
                WHERE user_id=? {deleted_clause}
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
    return [dict(r) for r in rows]


def soft_delete_conversations(user_id: int | None = None) -> int:
    """Soft-delete dialog rows (default clear path)."""
    with get_connection() as conn:
        if user_id is None:
            cur = conn.execute(
                """
                UPDATE conversations
                SET deleted_at=CURRENT_TIMESTAMP
                WHERE deleted_at IS NULL
                """
            )
        else:
            cur = conn.execute(
                """
                UPDATE conversations
                SET deleted_at=CURRENT_TIMESTAMP
                WHERE user_id=? AND deleted_at IS NULL
                """,
                (str(user_id),),
            )
        conn.commit()
        return int(cur.rowcount or 0)


def hard_delete_conversations(user_id: int | None = None) -> int:
    """Hard-delete dialog rows (explicit only)."""
    with get_connection() as conn:
        if user_id is None:
            cur = conn.execute("DELETE FROM conversations")
        else:
            cur = conn.execute(
                "DELETE FROM conversations WHERE user_id=?",
                (str(user_id),),
            )
        conn.commit()
        return int(cur.rowcount or 0)


def clear_conversations(user_id: int | None = None, *, hard: bool = False) -> int:
    """Clear memory: soft by default, hard when explicitly requested."""
    if hard:
        return hard_delete_conversations(user_id=user_id)
    return soft_delete_conversations(user_id=user_id)


def soft_delete_user(user_id: int) -> dict[str, int]:
    """Soft-delete user record and their conversations."""
    uid = str(user_id)
    with get_connection() as conn:
        ensure_user(user_id)
        cur_user = conn.execute(
            """
            UPDATE users
            SET deleted_at=CURRENT_TIMESTAMP
            WHERE user_id=? AND deleted_at IS NULL
            """,
            (uid,),
        )
        cur_conv = conn.execute(
            """
            UPDATE conversations
            SET deleted_at=CURRENT_TIMESTAMP
            WHERE user_id=? AND deleted_at IS NULL
            """,
            (uid,),
        )
        conn.commit()
    return {
        "users_marked": int(cur_user.rowcount or 0),
        "conversations_marked": int(cur_conv.rowcount or 0),
    }


def purge_expired_data(retention_days: int | None = None) -> dict[str, int]:
    """Hard-purge soft-deleted rows and usage events older than retention window."""
    days = get_retention_days() if retention_days is None else max(1, int(retention_days))
    with get_connection() as conn:
        cur_conv = conn.execute(
            """
            DELETE FROM conversations
            WHERE deleted_at IS NOT NULL
              AND deleted_at < datetime('now', ?)
            """,
            (f"-{days} days",),
        )
        cur_users = conn.execute(
            """
            DELETE FROM users
            WHERE deleted_at IS NOT NULL
              AND deleted_at < datetime('now', ?)
            """,
            (f"-{days} days",),
        )
        cur_usage = conn.execute(
            """
            DELETE FROM usage_events
            WHERE created_at < datetime('now', ?)
            """,
            (f"-{days} days",),
        )
        conn.commit()
    return {
        "conversations_purged": int(cur_conv.rowcount or 0),
        "users_purged": int(cur_users.rowcount or 0),
        "usage_events_purged": int(cur_usage.rowcount or 0),
        "retention_days": days,
    }


def record_usage_event(
    *,
    user_id: int,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    router_decision: str = "",
) -> int:
    ensure_user(user_id)
    decision = (router_decision or "").strip().lower()
    if decision not in {"search", "no_search"}:
        decision = "search" if decision in {"true", "1", "yes"} else "no_search"
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO usage_events (
                user_id, prompt_tokens, completion_tokens, estimated_cost_usd, router_decision
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(user_id),
                max(0, int(prompt_tokens)),
                max(0, int(completion_tokens)),
                max(0.0, float(estimated_cost_usd or 0.0)),
                decision,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def aggregate_usage_by_user(limit: int = 500) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 5000))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                user_id,
                COUNT(*) AS requests_count,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                MAX(created_at) AS last_request_at
            FROM usage_events
            GROUP BY user_id
            ORDER BY estimated_cost_usd DESC, requests_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def export_dialogs(
    *,
    user_id: int | None = None,
    format: str = "json",
    include_deleted: bool = False,
    limit: int = 5000,
) -> tuple[str, str, str]:
    """Return (body, media_type, filename) for dialog export."""
    rows = list_recent_conversations(
        limit=limit,
        user_id=user_id,
        include_deleted=include_deleted,
    )
    # Export chronological for readability.
    rows = list(reversed(rows))
    fmt = (format or "json").strip().lower()
    suffix = f"user_{user_id}" if user_id is not None else "all"
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["id", "user_id", "role", "content", "created_at", "deleted_at"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue(), "text/csv; charset=utf-8", f"dialogs_{suffix}.csv"
    body = json.dumps(rows, ensure_ascii=False, indent=2)
    return body, "application/json; charset=utf-8", f"dialogs_{suffix}.json"


def compute_async_queue_lag_seconds() -> float:
    """Max age in seconds of queued/running async_tasks (0 if none)."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT MAX(
                CAST(
                    (julianday('now') - julianday(created_at)) * 86400.0
                    AS REAL
                )
            ) AS lag_seconds
            FROM async_tasks
            WHERE lower(status) IN ('queued', 'running')
            """
        ).fetchone()
    if not row or row["lag_seconds"] is None:
        return 0.0
    return max(0.0, float(row["lag_seconds"]))


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
            SELECT user_id, created_at, last_seen, deleted_at
            FROM users
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_users() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE deleted_at IS NULL"
        ).fetchone()
    return int(row["c"] if row else 0)


def count_conversations() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM conversations WHERE deleted_at IS NULL"
        ).fetchone()
    return int(row["c"] if row else 0)


def count_user_requests() -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM conversations
            WHERE role='user' AND deleted_at IS NULL
            """
        ).fetchone()
    return int(row["c"] if row else 0)


def count_error_responses() -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM conversations
            WHERE role='assistant'
              AND deleted_at IS NULL
              AND content LIKE '%Произошла ошибка%'
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
