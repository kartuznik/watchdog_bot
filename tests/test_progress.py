"""Tests for Telegram progress reporter."""

from __future__ import annotations

import pytest

from telegram_bot.progress import ProgressReporter, stage_for_node, stage_label


class _FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self.deleted: list[dict] = []
        self._next_id = 100

    async def send_message(self, *, chat_id: int, text: str):
        self._next_id += 1
        self.sent.append({"chat_id": chat_id, "text": text, "message_id": self._next_id})
        return type("Msg", (), {"message_id": self._next_id})()

    async def edit_message_text(self, *, chat_id: int, message_id: int, text: str):
        self.edited.append(
            {"chat_id": chat_id, "message_id": message_id, "text": text}
        )

    async def delete_message(self, *, chat_id: int, message_id: int):
        self.deleted.append({"chat_id": chat_id, "message_id": message_id})


class _ExplodingDeleteBot(_FakeBot):
    async def delete_message(self, *, chat_id: int, message_id: int):
        raise RuntimeError("message to delete not found")


@pytest.mark.asyncio
async def test_progress_reporter_updates_and_skips_same_stage() -> None:
    bot = _FakeBot()
    progress = ProgressReporter(bot, chat_id=42)
    mid = await progress.start("routing")
    assert mid == 101
    assert bot.sent[0]["text"] == stage_label("routing")

    await progress.set_stage("search")
    await progress.set_stage("search")  # spam protection
    await progress.set_stage("writing")
    assert len(bot.edited) == 2
    assert bot.edited[0]["text"] == stage_label("search")
    assert bot.edited[1]["text"] == stage_label("writing")

    await progress.finish(mode="delete")
    assert bot.deleted == [{"chat_id": 42, "message_id": 101}]
    assert progress.message_id is None


@pytest.mark.asyncio
async def test_progress_finish_swallows_telegram_errors() -> None:
    bot = _ExplodingDeleteBot()
    progress = ProgressReporter(bot, chat_id=7)
    await progress.start("routing")
    await progress.finish(mode="delete")  # must not raise
    assert progress.message_id is None


def test_stage_for_node_mapping() -> None:
    assert stage_for_node("router_node") == "routing"
    assert stage_for_node("web_search_node") == "search"
    assert stage_for_node("writer_node") == "writing"
    assert stage_for_node("unknown") is None
