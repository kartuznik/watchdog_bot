"""Search query reformulation and source relevance filter tests."""

from __future__ import annotations

import logging

from agents.web_search import build_search_query, filter_relevant_sources, significant_terms


def test_interrogative_news_query_keeps_topic_terms() -> None:
    original = "какие новости про ИИ сегодня"
    query = build_search_query(original)
    # Must not collapse to the interrogative alone.
    assert query.strip().lower() != "какие"
    assert "какие" not in query.lower().split()
    sig = significant_terms(query)
    assert "новости" in sig
    assert "сегодня" in sig
    # "ИИ" / "ии" must survive as a topic signal.
    assert "ии" in sig
    # Gate must keep topic signal from the original.
    assert significant_terms(original) & sig


def test_build_search_query_what_when_how_class() -> None:
    cases = [
        "что случилось с OpenAI сегодня",
        "как изменился рынок GPU в 2026",
        "где смотреть новости про нейросети",
    ]
    for original in cases:
        query = build_search_query(original)
        assert len(query) >= 8
        assert significant_terms(original) & significant_terms(query)
        first = query.split()[0].lower()
        assert first not in {"что", "как", "где", "какие"}


def test_search_query_gate_falls_back_to_original(caplog: logging.LogCaptureFixture) -> None:
    original = "ИИ"
    with caplog.at_level(logging.WARNING):
        query = build_search_query(original, min_chars=8)
    assert query == original
    assert any("search query gate" in r.message for r in caplog.records)


def test_filter_drops_dictionary_titles_for_news_query() -> None:
    query = "какие новости про ИИ сегодня"
    sources = [
        {
            "title": "какие - English translation – Linguee",
            "url": "https://www.linguee.com/russian-english/translation/какие.html",
        },
        {
            "title": "Новости ИИ: свежие релизы нейросетей сегодня",
            "url": "https://example.com/ai-news-today",
        },
        {
            "title": "Dictionary: какие meaning",
            "url": "https://dictionary.example/какие",
        },
    ]
    kept = filter_relevant_sources(query, sources)
    urls = {item["url"] for item in kept}
    assert "https://example.com/ai-news-today" in urls
    assert all("linguee" not in u for u in urls)
    assert all("dictionary.example" not in u for u in urls)
