"""Grounded generation: evidence cards, freshness gate, mock writer/reviewer."""

from __future__ import annotations

from agents.multi_agent import (
    _mock_review,
    _mock_writer,
    build_initial_multi_agent_state,
    review_grounding_freshness,
    topic_needs_freshness,
)
from agents.web_search import format_evidence_cards, normalize_source_items


def _fresh_evidence():
    return normalize_source_items(
        [
            {
                "title": "OpenAI launches Widget API",
                "url": "https://example.com/openai-widget-2026",
                "snippet": (
                    "5 августа 2026 года OpenAI представила Widget API для ChatGPT, "
                    "позволяющий встраивать интерактивные виджеты партнёров."
                ),
                "published_at": "2026-08-05",
            },
            {
                "title": "EU AI Act enforcement update",
                "url": "https://example.com/eu-ai-act-2026",
                "snippet": (
                    "В июле 2026 регуляторы ЕС уточнили сроки применения AI Act "
                    "для высокорисковых систем."
                ),
                "published_at": "2026-07-18",
            },
        ]
    )


def test_topic_needs_freshness_markers() -> None:
    assert topic_needs_freshness("какие новости про ИИ сегодня")
    assert topic_needs_freshness("latest AI breaking news")
    assert not topic_needs_freshness("что такое градиентный спуск")


def test_mock_writer_reflects_fresh_snippet_dates() -> None:
    evidence = _fresh_evidence()
    state = build_initial_multi_agent_state(
        topic="какие новости про ИИ сегодня",
        user_id=1,
        use_llm=False,
    )
    state["source_evidence"] = evidence
    state["web_sources"] = evidence
    state["research_data"] = format_evidence_cards(evidence)
    draft = _mock_writer(state)["draft"]
    assert "2026-08-05" in draft
    assert "Widget API" in draft or "виджет" in draft.lower()
    assert "2023" not in draft


def test_reviewer_rejects_memory_dates_not_in_evidence() -> None:
    evidence = _fresh_evidence()
    needs_revise, feedback = review_grounding_freshness(
        "какие новости про ИИ сегодня",
        "В октябре 2023 года в России запустили новый ИИ-сервис для медиков.",
        evidence,
    )
    assert needs_revise is True
    assert "2023" in feedback or "рассинхрон" in feedback.lower() or "дат" in feedback.lower()


def test_mock_review_revises_ungrounded_freshness_draft() -> None:
    evidence = _fresh_evidence()
    state = build_initial_multi_agent_state(
        topic="какие новости про ИИ сегодня",
        user_id=1,
        use_llm=False,
    )
    state["source_evidence"] = evidence
    state["web_sources"] = evidence
    state["draft"] = (
        "В октябре 2023 года искусственный интеллект продолжал набирать популярность."
    )
    result = _mock_review(state)
    assert result.get("feedback")
    assert int(result.get("revision_count", 0)) == 1


def test_format_evidence_cards_include_snippet_and_date() -> None:
    cards = format_evidence_cards(_fresh_evidence())
    assert "Дата публикации: 2026-08-05" in cards
    assert "Widget API" in cards
    assert "URL: https://example.com/openai-widget-2026" in cards
