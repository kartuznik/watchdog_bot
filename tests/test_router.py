"""Tests for smart query router."""

from __future__ import annotations

from agents.metrics import agent_router_decisions_total, observe_router_decision
from agents.multi_agent import route_after_router, run_multi_agent
from agents.router import heuristic_route, merge_router_decision, parse_router_payload


def test_heuristic_creative_skips_search() -> None:
    need, mode, reason = heuristic_route("Напиши стих про кота")
    assert need is False
    assert mode == "creative"
    assert "creative" in reason


def test_heuristic_news_needs_search() -> None:
    need, mode, reason = heuristic_route("Какие новости про ИИ сегодня?")
    assert need is True
    assert mode == "research"
    assert "search" in reason


def test_heuristic_definition_no_search() -> None:
    need, mode, _reason = heuristic_route("Что такое энтропия? Объясни термин")
    assert need is False
    assert mode == "factual"


def test_parse_router_payload() -> None:
    parsed = parse_router_payload(
        '{"need_web_search": false, "mode": "creative", "reason": "poem"}'
    )
    assert parsed == (False, "creative", "poem")


def test_merge_router_respects_web_search_flag() -> None:
    decision = merge_router_decision(
        topic="Новости рынка сегодня",
        llm_raw='{"need_web_search": true, "mode": "research", "reason": "news"}',
        web_search_enabled=False,
    )
    assert decision["need_web_search"] is False
    assert decision["router_decision"] == "no_search"
    assert "web_search_disabled" in decision["router_reason"]


def test_route_after_router_branches() -> None:
    assert (
        route_after_router(
            {
                "need_web_search": True,
                "router_mode": "research",
            }  # type: ignore[arg-type]
        )
        == "web_search_node"
    )
    assert (
        route_after_router(
            {
                "need_web_search": False,
                "router_mode": "creative",
            }  # type: ignore[arg-type]
        )
        == "writer_node"
    )
    assert (
        route_after_router(
            {
                "need_web_search": False,
                "router_mode": "factual",
            }  # type: ignore[arg-type]
        )
        == "research_node"
    )


def test_mock_graph_router_creative_path() -> None:
    result = run_multi_agent("напиши стих про море", use_llm=False)
    assert result["need_web_search"] is False
    assert result["router_mode"] == "creative"
    assert result["draft"]
    assert result["web_sources"] == []


def test_observe_router_decision_increments() -> None:
    before = agent_router_decisions_total.labels(decision="no_search")._value.get()
    observe_router_decision("no_search")
    after = agent_router_decisions_total.labels(decision="no_search")._value.get()
    assert after == before + 1
