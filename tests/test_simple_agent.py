"""Tests for simple LangGraph agent."""

from agents.simple_agent import run_simple_agent


def test_math_query_routes_to_math_node_and_returns_result() -> None:
    state = run_simple_agent("2+2=?", classifier_fn=lambda _: "math")

    assert state["classification"] == "math"
    assert state["tool_result"] == "Math result: 4"


def test_code_query_routes_to_code_node_and_returns_snippet() -> None:
    state = run_simple_agent(
        "напиши функцию hello world на python",
        classifier_fn=lambda _: "code",
    )

    assert state["classification"] == "code"
    assert "def hello_world()" in state["tool_result"]


def test_general_query_routes_to_general_node() -> None:
    state = run_simple_agent("привет, как дела?", classifier_fn=lambda _: "general")

    assert state["classification"] == "general"
    assert state["tool_result"].startswith("General response:")
