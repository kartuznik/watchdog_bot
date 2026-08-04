"""Tests for product Telegram presentation (Phase 1.5)."""

from __future__ import annotations

from telegram_bot.messaging import (
    CONTINUATION_PREFIX,
    CONTINUATION_SUFFIX,
    TELEGRAM_SAFE_CHUNK,
    build_result_messages,
    build_sources_keyboard,
    chunk_text,
    format_sources_list_html,
    format_user_facing_error,
)


def _sample_result(**overrides):
    base = {
        "topic": "решение теории пуанкаре",
        "draft": "Гипотеза Пуанкаре доказана Перельманом через поток Риччи.",
        "research_summary": (
            "1. Формулировка касается трёхмерных многообразий.\n"
            "2. Доказательство опирается на поток Риччи.\n"
            "3. Практический вывод: топология получила новый инструмент."
        ),
        "research_data": "длинное сырое исследование " * 40,
        "web_sources": [
            {"title": "Статья о Перельмане", "url": "https://example.com/perelman"},
            {"title": "Обзор потока Риччи", "url": "https://example.com/ricci"},
        ],
        "revision_count": 1,
        "llm_prompt_tokens": 100,
        "llm_completion_tokens": 50,
        "estimated_cost_usd": 0.001,
    }
    base.update(overrides)
    return base


def test_chunk_text_splits_on_sentence_boundary_not_mid_word() -> None:
    sentence = "Это законченное предложение про Пуанкаре. "
    text = sentence * 200
    assert len(text) > 4096
    parts = chunk_text(text, max_len=800)
    assert len(parts) >= 2
    assert all(len(part) <= 800 for part in parts)
    for part in parts[:-1]:
        assert "продолжение ниже" in part
        body = part
        if body.startswith(CONTINUATION_PREFIX):
            body = body[len(CONTINUATION_PREFIX) :]
        if body.endswith(CONTINUATION_SUFFIX):
            body = body[: -len(CONTINUATION_SUFFIX)]
        body = body.rstrip()
        # Sentence-boundary chunking: non-final bodies end with sentence punctuation.
        assert body.endswith((".", "!", "?", "…"))
    assert "продолжение" in parts[1]


def test_chunk_text_short_message_unchanged() -> None:
    assert chunk_text("короткий ответ") == ["короткий ответ"]
    assert chunk_text("") == []
    assert TELEGRAM_SAFE_CHUNK == 3500


def test_message_order_draft_then_research_then_sources() -> None:
    messages = build_result_messages(_sample_result(), viewer_role="user")
    sections = [m.section for m in messages]
    assert sections[0] == "draft"
    assert "research" in sections
    assert "sources" in sections
    assert sections.index("draft") < sections.index("research") < sections.index("sources")
    assert "📝" in messages[0].text
    assert "🔬" in next(m.text for m in messages if m.section == "research")
    assert "🔗" in next(m.text for m in messages if m.section == "sources")
    assert "###" not in "".join(m.text for m in messages)


def test_footer_hidden_for_user_visible_for_admin_and_owner() -> None:
    user_msgs = build_result_messages(_sample_result(), viewer_role="user")
    admin_msgs = build_result_messages(_sample_result(), viewer_role="admin")
    owner_msgs = build_result_messages(_sample_result(), viewer_role="owner")
    user_text = "\n".join(m.text for m in user_msgs)
    admin_text = "\n".join(m.text for m in admin_msgs)
    owner_text = "\n".join(m.text for m in owner_msgs)
    assert "tokens≈" not in user_text
    assert "cost≈" not in user_text
    assert "tokens≈" in admin_text
    assert "cost≈" in owner_text


def test_sources_numbered_list_and_inline_keyboard() -> None:
    messages = build_result_messages(_sample_result(), viewer_role="user")
    sources_msg = next(m for m in messages if m.section == "sources")
    assert "1." in sources_msg.text
    assert "Статья о Перельмане" in sources_msg.text
    assert "Источник:" not in sources_msg.text
    # Clickable HTML anchors with full article URLs (not bare domains).
    assert 'href="https://example.com/perelman"' in sources_msg.text
    assert 'href="https://example.com/ricci"' in sources_msg.text
    assert "(example.com)" not in sources_msg.text
    assert sources_msg.reply_markup is not None
    keyboard = sources_msg.reply_markup
    assert len(keyboard.inline_keyboard) == 2
    assert keyboard.inline_keyboard[0][0].url == "https://example.com/perelman"
    assert keyboard.inline_keyboard[1][0].url == "https://example.com/ricci"

    encoded = "https://example.com/path%20with%20space?q=1&x=2"
    html_list = format_sources_list_html(
        [{"title": "Title <script>& more", "url": encoded}]
    )
    assert 'href="https://example.com/path%20with%20space?q=1&amp;x=2"' in html_list
    assert "<b>Title &lt;script&gt;&amp; more</b>" in html_list
    assert "(example.com)" not in html_list
    assert html_list.startswith('1. <a href="')
    kb = build_sources_keyboard(_sample_result()["web_sources"])
    assert kb is not None
    long_label_kb = build_sources_keyboard(
        [{"title": "A" * 80, "url": "https://example.com/long"}]
    )
    assert long_label_kb is not None
    assert len(long_label_kb.inline_keyboard[0][0].text) <= 64
    assert long_label_kb.inline_keyboard[0][0].text.endswith("…")


def test_html_escapes_angle_brackets_in_draft() -> None:
    messages = build_result_messages(
        _sample_result(draft="A <script>alert(1)</script> & B"),
        viewer_role="user",
    )
    draft = messages[0].text
    assert "<script>" not in draft
    assert "&lt;script&gt;" in draft
    assert "&amp;" in draft


def test_format_user_facing_error_message_too_long() -> None:
    msg = format_user_facing_error(
        RuntimeError("Telegram server says - Bad Request: message is too long")
    )
    assert "длинн" in msg.lower()
    assert "sk-" not in msg
