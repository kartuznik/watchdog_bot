"""Tests for week 2 multi-agent loop workflow."""

from agents.multi_agent import run_multi_agent


def test_reviewer_feedback_loop_stops_with_iteration_limit() -> None:
    result = run_multi_agent("плохой черновик")

    assert result["user_id"] == 0
    assert result["topic"] == "плохой черновик"
    assert isinstance(result["conversation_history"], list)
    assert result["use_llm"] is False
    assert result["research_data"]
    assert result["draft"]

    # Reviewer should request at least one revision.
    assert result["revision_count"] >= 1

    # Final draft should be accepted, so feedback is cleared and graph ends.
    assert result["feedback"] == ""
    assert "отлично" in result["draft"].lower()

    # Safety guarantee against infinite loop.
    assert result["revision_count"] <= 2
