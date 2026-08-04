"""Tests for Telegram chunking and honest error mapping."""

from __future__ import annotations

from telegram_bot.messaging import (
    TELEGRAM_SAFE_CHUNK,
    build_result_message_parts,
    chunk_text,
    format_user_facing_error,
)


def test_chunk_text_splits_long_answer_under_telegram_limit() -> None:
    text = ("Параграф про теорию Пуанкаре. " * 40 + "\n\n") * 8
    assert len(text) > 4096
    parts = chunk_text(text, max_len=3800)
    assert len(parts) >= 2
    assert all(len(part) <= 3800 for part in parts)
    assert all(len(part) <= TELEGRAM_SAFE_CHUNK for part in parts)
    joined = "".join(parts)
    assert "Пуанкаре" in joined
    assert "Параграф" in joined
    assert abs(len(joined) - len(text.strip())) <= len(parts) * 2


def test_chunk_text_short_message_unchanged() -> None:
    assert chunk_text("короткий ответ") == ["короткий ответ"]
    assert chunk_text("") == []


def test_build_result_message_parts_draft_first_and_chunked() -> None:
    long_research = ("Факт о многообразиях. " * 100).strip()
    long_draft = ("Решение гипотезы Пуанкаре вкратце. " * 80).strip()
    result = {
        "topic": "решение теории пуанкаре",
        "research_data": long_research,
        "draft": long_draft,
        "web_sources": ["https://example.com/poincare"],
        "revision_count": 1,
        "llm_prompt_tokens": 10,
        "llm_completion_tokens": 20,
        "estimated_cost_usd": 0.0001,
        "user_id": 1,
        "conversation_history": [],
        "feedback": "",
        "use_llm": False,
    }
    parts = build_result_message_parts(result)  # type: ignore[arg-type]
    assert parts
    assert parts[0].startswith("✅ Готово")
    assert "### Draft" in parts[0]
    assert any("Research" in part for part in parts)
    assert any("Источник: [https://example.com/poincare]" in part for part in parts)
    assert all(len(part) <= 3800 for part in parts)


def test_format_user_facing_error_message_too_long() -> None:
    msg = format_user_facing_error(RuntimeError("Telegram server says - Bad Request: message is too long"))
    assert "длинн" in msg.lower()
    assert "sk-" not in msg


def test_format_user_facing_error_tavily() -> None:
    class InvalidAPIKeyError(Exception):
        pass

    msg = format_user_facing_error(InvalidAPIKeyError("Unauthorized: missing or invalid API key."))
    assert "Tavily" in msg
    assert "tvly-" not in msg
