"""Anchor storage adapted from telegram-ai-bot user_anchors module."""

from __future__ import annotations

from agents.database import get_connection

MAX_ANCHOR_TITLE_LEN = 200
MAX_SNIPPET_LEN = 1000


def auto_title_anchor(snippet: str, fallback: str = "Якорь") -> str:
    line = (snippet or "").strip().split("\n", 1)[0].strip()
    line = " ".join(line.split())
    if not line:
        return fallback[:MAX_ANCHOR_TITLE_LEN]
    if len(line) > 72:
        line = line[:69] + "..."
    return line[:MAX_ANCHOR_TITLE_LEN]


def create_anchor(user_id: int, title: str, context_snippet: str, message_ref: int = 0) -> int:
    safe_title = (title or "Якорь").strip()[:MAX_ANCHOR_TITLE_LEN]
    safe_snippet = (context_snippet or "").strip()[:MAX_SNIPPET_LEN]
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversation_anchors (user_id, title, context_snippet, message_ref)
            VALUES (?, ?, ?, ?)
            """,
            (str(user_id), safe_title, safe_snippet, int(message_ref)),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_user_anchors(user_id: int, limit: int = 50) -> list[dict]:
    limit = max(1, min(limit, 200))
    with get_connection() as conn:
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


def delete_anchor(anchor_id: int, user_id: int | None = None) -> int:
    with get_connection() as conn:
        if user_id is None:
            cur = conn.execute("DELETE FROM conversation_anchors WHERE id=?", (int(anchor_id),))
        else:
            cur = conn.execute(
                "DELETE FROM conversation_anchors WHERE id=? AND user_id=?",
                (int(anchor_id), str(user_id)),
            )
        conn.commit()
        return int(cur.rowcount or 0)
