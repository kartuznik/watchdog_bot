"""Telegram progress reporter for multi-agent pipeline stages."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Graph node name -> logical stage key
NODE_STAGE_MAP: dict[str, str] = {
    "router_node": "routing",
    "web_search_node": "search",
    "research_node": "research",
    "research_summary_node": "research",
    "writer_node": "writing",
    "reviewer_node": "review",
}

STAGE_LABELS: dict[str, str] = {
    "queued": "⏳ Задача в очереди…",
    "routing": "🧭 Определяю маршрут…",
    "search": "🔎 Ищу источники…",
    "research": "🧠 Анализирую данные…",
    "writing": "✍️ Формирую ответ…",
    "review": "🧪 Проверяю качество…",
    "working": "⚙️ Обрабатываю запрос…",
    "done": "✅ Готово",
}


def stage_label(stage: str) -> str:
    key = (stage or "").strip().lower()
    return STAGE_LABELS.get(key, STAGE_LABELS["working"])


def stage_for_node(node_name: str) -> str | None:
    return NODE_STAGE_MAP.get(node_name)


class ProgressReporter:
    """Single service message updated as pipeline stages complete."""

    def __init__(self, bot: Any, chat_id: int) -> None:
        self.bot = bot
        self.chat_id = int(chat_id)
        self.message_id: int | None = None
        self._last_stage: str | None = None
        self._last_text: str | None = None

    async def start(self, stage: str = "routing") -> int | None:
        text = stage_label(stage)
        try:
            msg = await self.bot.send_message(chat_id=self.chat_id, text=text)
            self.message_id = int(getattr(msg, "message_id", 0) or 0) or None
            self._last_stage = stage
            self._last_text = text
            return self.message_id
        except Exception:
            logger.exception("ProgressReporter.start failed chat_id=%s", self.chat_id)
            self.message_id = None
            return None

    async def set_stage(self, stage: str) -> None:
        key = (stage or "").strip().lower() or "working"
        if key == self._last_stage:
            return  # spam protection: no edit when stage unchanged
        text = stage_label(key)
        if text == self._last_text and self.message_id is not None:
            self._last_stage = key
            return
        self._last_stage = key
        self._last_text = text
        if self.message_id is None:
            await self.start(key)
            return
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self.message_id,
                text=text,
            )
        except Exception as exc:
            # Ignore "message is not modified" and similar benign Telegram errors.
            msg = str(exc).lower()
            if "message is not modified" in msg or "message to edit not found" in msg:
                logger.info("Progress edit skipped: %s", type(exc).__name__)
                return
            logger.warning("ProgressReporter.set_stage failed: %s", exc)

    async def finish(self, *, mode: str = "delete") -> None:
        """Delete progress message or edit it into a short header. Never raise."""
        if self.message_id is None:
            return
        try:
            if mode == "header":
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.message_id,
                    text=stage_label("done"),
                )
            else:
                await self.bot.delete_message(
                    chat_id=self.chat_id,
                    message_id=self.message_id,
                )
        except Exception as exc:
            logger.info(
                "ProgressReporter.finish ignored error=%s mode=%s",
                type(exc).__name__,
                mode,
            )
        finally:
            self.message_id = None
            self._last_stage = None
            self._last_text = None
