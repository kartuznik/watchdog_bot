"""Tests for multi-agent review loop, sources attachment, and history wiring."""

from __future__ import annotations

from agents.multi_agent import (
    build_initial_multi_agent_state,
    ensure_sources_block,
    parse_reviewer_payload,
    run_multi_agent,
    should_approve_review,
)
from worker import _normalize_history


def test_reviewer_feedback_loop_stops_with_iteration_limit() -> None:
    result = run_multi_agent("плохой черновик")

    assert result["user_id"] == 0
    assert result["topic"] == "плохой черновик"
    assert isinstance(result["conversation_history"], list)
    assert result["use_llm"] is False
    assert result["research_data"]
    assert result["draft"]
    assert result["revision_count"] >= 1
    assert result["feedback"] == ""
    assert "отлично" not in result["draft"].lower()
    assert "улучшенный материал" in result["draft"].lower()
    assert result["revision_count"] <= 2


def test_good_topic_approves_without_revision() -> None:
    result = run_multi_agent("обзор агентов")
    assert result["feedback"] == ""
    assert result["revision_count"] == 0
    assert "отлично" not in result["draft"].lower()


def test_ensure_sources_block_appends_missing_urls() -> None:
    draft = "Короткий ответ без ссылок."
    sources = ["https://example.com/a", "https://example.com/b"]
    out = ensure_sources_block(draft, sources)
    assert "### Источники" in out
    assert "Источник: [https://example.com/a]" in out
    assert "Источник: [https://example.com/b]" in out


def test_ensure_sources_block_noop_when_already_present() -> None:
    draft = (
        "Ответ.\n\n### Источники\n"
        "Источник: [https://example.com/a]\n"
        "Источник: [https://example.com/b]"
    )
    sources = ["https://example.com/a", "https://example.com/b"]
    assert ensure_sources_block(draft, sources) == draft


def test_parse_reviewer_payload_structured_json() -> None:
    raw = """
    {
      "decision": "revise",
      "scores": {"clarity": 4, "completeness": 2, "grounding": 3, "structure": 2},
      "feedback": "Добавь выводы"
    }
    """
    decision, scores, feedback = parse_reviewer_payload(raw)
    assert decision == "revise"
    assert scores["completeness"] == 2.0
    assert feedback == "Добавь выводы"
    assert should_approve_review(decision, scores, feedback) is False


def test_parse_reviewer_payload_approve_with_high_scores() -> None:
    raw = (
        '{"decision":"approve","scores":'
        '{"clarity":5,"completeness":4,"grounding":4,"structure":5},"feedback":""}'
    )
    decision, scores, feedback = parse_reviewer_payload(raw)
    assert should_approve_review(decision, scores, feedback) is True


def test_worker_normalizes_conversation_history() -> None:
    history = _normalize_history(
        [
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "ответ"},
            {"role": "system", "content": "skip"},
            {"role": "user", "content": ""},
            "broken",
        ]
    )
    assert history == [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "ответ"},
    ]


def test_initial_state_keeps_conversation_history() -> None:
    history = [{"role": "user", "content": "контекст"}]
    state = build_initial_multi_agent_state(
        topic="тема",
        user_id=42,
        conversation_history=history,
        use_llm=False,
    )
    assert state["conversation_history"] == history
    assert state["estimated_cost_usd"] == 0.0
