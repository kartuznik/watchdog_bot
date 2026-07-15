"""Conversation memory adapted from telegram-ai-bot ChatMemory."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from agents.database import ensure_user, get_connection


class ChatMemory:
    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._store: Dict[int, List[dict[str, str]]] = defaultdict(list)

    def add(self, user_id: int, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            return
        try:
            ensure_user(user_id)
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
                    (str(user_id), role, content),
                )
                conn.execute(
                    "UPDATE users SET last_seen=CURRENT_TIMESTAMP WHERE user_id=?",
                    (str(user_id),),
                )
                conn.commit()
        except Exception:
            self._store[user_id].append({"role": role, "content": content})
            if len(self._store[user_id]) > self.max_messages:
                self._store[user_id] = self._store[user_id][-self.max_messages :]

    def get(self, user_id: int) -> List[dict[str, str]]:
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT id, role, content
                        FROM conversations
                        WHERE user_id=?
                        ORDER BY id DESC
                        LIMIT ?
                    ) t
                    ORDER BY id ASC
                    """,
                    (str(user_id), self.max_messages),
                ).fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]
        except Exception:
            return list(self._store[user_id])

    def save_user_memory(self, user_id: int, user_msg: str, bot_msg: str) -> None:
        self.add(user_id, "user", user_msg)
        self.add(user_id, "assistant", bot_msg)

    def get_user_memory(self, user_id: int) -> List[dict[str, str]]:
        return self.get(user_id)

    def clear_user_memory(self, user_id: int) -> int:
        try:
            with get_connection() as conn:
                cur = conn.execute(
                    "DELETE FROM conversations WHERE user_id=?",
                    (str(user_id),),
                )
                conn.commit()
                return int(cur.rowcount or 0)
        except Exception:
            deleted = len(self._store[user_id])
            self._store[user_id].clear()
            return deleted
