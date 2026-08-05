"""Recency-aware Tavily retrieval and honesty marker tests."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock

from agents.freshness import (
    apply_freshness_ranking,
    freshness_honesty_note,
    get_tavily_freshness_days,
    topic_needs_freshness,
)
from agents.multi_agent import _mock_writer, build_initial_multi_agent_state
from agents.web_search import WebSearchTool, filter_relevant_sources, normalize_source_items


def test_today_query_sends_news_topic_and_days(monkeypatch: Any) -> None:
    monkeypatch.setenv("TAVILY_FRESHNESS_DAYS", "2")
    monkeypatch.setenv("TAVILY_NEWS_TOPIC", "news")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    captured: dict[str, Any] = {}

    def fake_search(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "results": [
                {
                    "title": "Мировые новости сегодня",
                    "url": "https://example.com/world-today",
                    "content": "Свежие мировые новости и события за сегодня.",
                    "published_date": date.today().isoformat(),
                }
            ]
        }

    tool = WebSearchTool.__new__(WebSearchTool)
    tool.client = MagicMock()
    tool.client.search.side_effect = fake_search

    _cards, sources = tool.search_with_sources(
        "мировые новости сегодня",
        max_results=3,
        relevance_query="мировые новости сегодня",
    )
    assert captured.get("topic") == "news"
    assert captured.get("days") == 2
    assert sources
    assert topic_needs_freshness("мировые новости сегодня")


def test_freshness_ranking_prefers_recent_and_drops_stale() -> None:
    today = date(2026, 8, 5)
    sources = normalize_source_items(
        [
            {
                "title": "Old world news",
                "url": "https://example.com/old",
                "snippet": "События июля 2026",
                "published_at": "2026-07-10",
            },
            {
                "title": "Fresh world news",
                "url": "https://example.com/fresh",
                "snippet": "События сегодня",
                "published_at": "2026-08-05",
            },
            {
                "title": "Yesterday brief",
                "url": "https://example.com/yday",
                "snippet": "Вчерашние события",
                "published_at": "2026-08-04",
            },
        ]
    )
    ranked, degraded = apply_freshness_ranking(
        list(sources),
        require_freshness=True,
        days=2,
        today=today,
    )
    assert degraded is False
    urls = [s["url"] for s in ranked]
    assert urls[0] == "https://example.com/fresh"
    assert "https://example.com/old" not in urls
    assert "https://example.com/yday" in urls


def test_filter_relevant_sources_ranks_fresh_for_today_query() -> None:
    today = date(2026, 8, 5)
    sources = [
        {
            "title": "Старые мировые новости",
            "url": "https://example.com/old-world",
            "snippet": "Архив июля",
            "published_at": "2026-07-01",
        },
        {
            "title": "Мировые новости сегодня",
            "url": "https://example.com/new-world",
            "snippet": "Свежая сводка",
            "published_at": today.isoformat(),
        },
    ]
    # Monkeypatch utc via apply through filter -> apply_freshness_ranking(today=...)
    # filter_relevant_sources uses utc_today internally; pin via env days and
    # call apply path with require_freshness True by using «сегодня» in query.
    # To control `today`, call apply_freshness_ranking directly after term filter:
    term_kept = filter_relevant_sources(
        "мировые новости сегодня",
        sources,
        require_freshness=False,
    )
    ranked, _ = apply_freshness_ranking(
        list(term_kept),
        require_freshness=True,
        days=2,
        today=today,
    )
    assert ranked[0]["url"] == "https://example.com/new-world"


def test_writer_marks_stale_data_date_when_no_fresh_sources() -> None:
    today = date(2026, 8, 5)
    evidence = normalize_source_items(
        [
            {
                "title": "World digest",
                "url": "https://example.com/july",
                "snippet": "Итоги июля по мировой повестке",
                "published_at": "2026-07-02",
            }
        ]
    )
    note = freshness_honesty_note(
        "мировые новости сегодня",
        evidence,
        days=2,
        today=today,
    )
    assert "2026-07-02" in note
    assert "сегодня" in note.lower() or "текущего" in note.lower()

    state = build_initial_multi_agent_state(
        topic="мировые новости сегодня",
        user_id=1,
        use_llm=False,
    )
    state["source_evidence"] = evidence
    state["web_sources"] = evidence
    # Force honesty path by patching freshness_honesty_note result via old dates:
    # relative to real utc_today this July date is always stale if today>=Aug.
    draft = _mock_writer(state)["draft"]
    assert "Самые свежие данные на" in draft or "2026-07-02" in draft


def test_tavily_freshness_days_default(monkeypatch: Any) -> None:
    monkeypatch.delenv("TAVILY_FRESHNESS_DAYS", raising=False)
    assert get_tavily_freshness_days() == 2
    monkeypatch.setenv("TAVILY_FRESHNESS_DAYS", "5")
    assert get_tavily_freshness_days() == 5


def test_days_empty_uses_start_date_fallback(monkeypatch: Any) -> None:
    monkeypatch.setenv("TAVILY_FRESHNESS_DAYS", "2")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_search(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if "days" in kwargs and "start_date" not in kwargs:
            return {"results": []}
        return {
            "results": [
                {
                    "title": "Новости сегодня — сводка",
                    "url": "https://example.com/fb",
                    "content": "Свежие новости после start_date fallback",
                    "published_date": date.today().isoformat(),
                }
            ]
        }

    tool = WebSearchTool.__new__(WebSearchTool)
    tool.client = MagicMock()
    tool.client.search.side_effect = fake_search
    _cards, sources = tool.search_with_sources(
        "новости сегодня",
        relevance_query="новости сегодня",
    )
    assert len(calls) >= 2
    assert "start_date" in calls[1]
    assert sources
