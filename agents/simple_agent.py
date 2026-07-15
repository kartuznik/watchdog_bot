"""Simple LangGraph agent implementation for week 1."""

from __future__ import annotations

import ast
import logging
import re
from typing import Callable, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.llm_config import LLMConfig

logger = logging.getLogger(__name__)

ClassificationLabel = Literal["math", "code", "general"]


class AgentState(TypedDict):
    """Shared state that flows through the LangGraph nodes."""

    messages: list[str]
    classification: ClassificationLabel
    tool_result: str


ClassifierFn = Callable[[str], ClassificationLabel]


def _normalize_label(raw_label: str) -> ClassificationLabel:
    normalized = raw_label.strip().lower()
    if normalized in {"math", "code", "general"}:
        return cast(ClassificationLabel, normalized)
    if "math" in normalized:
        return "math"
    if "code" in normalized:
        return "code"
    return "general"


def _classify_with_llm(user_query: str) -> ClassificationLabel:
    """Classify a request using configured provider/model."""
    provider = LLMConfig.get_provider()
    model_name = LLMConfig.get_model_name()
    logger.info("Using provider: %s, model: %s", provider, model_name)
    model = LLMConfig.create_chat_model(temperature=0)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict query classifier. "
                    "Return exactly one label: math, code, or general."
                )
            ),
            HumanMessage(content=user_query),
        ]
    )
    return _normalize_label(str(response.content))


def _safe_eval_math(expression: str) -> float:
    """Evaluate a simple arithmetic expression safely via AST."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
        raise ValueError("Unsupported expression")

    cleaned_source = (
        expression.replace("=", " ")
        .replace("?", " ")
        .replace(",", ".")
        .replace("×", "*")
    )
    candidates = [
        chunk.strip()
        for chunk in re.findall(r"[\d\.\s\+\-\*\/\(\)\^]+", cleaned_source)
    ]
    math_candidates = [
        chunk
        for chunk in candidates
        if any(char.isdigit() for char in chunk)
        and any(op in chunk for op in "+-*/^")
    ]
    cleaned = max(math_candidates, key=len) if math_candidates else cleaned_source.strip()
    cleaned = cleaned.replace("^", "**")
    parsed = ast.parse(cleaned, mode="eval")
    return _eval(parsed)


def _route_by_classification(state: AgentState) -> ClassificationLabel:
    return _normalize_label(state.get("classification", "general"))


def build_simple_agent_graph(classifier_fn: ClassifierFn | None = None):
    """Build and compile the simple routing graph."""
    classifier = classifier_fn or _classify_with_llm
    graph = StateGraph(AgentState)

    def classify_node(state: AgentState) -> dict[str, ClassificationLabel]:
        user_query = state["messages"][-1] if state["messages"] else ""
        return {"classification": classifier(user_query)}

    def math_node(state: AgentState) -> dict[str, str]:
        user_query = state["messages"][-1] if state["messages"] else ""
        try:
            result = _safe_eval_math(user_query)
            if result.is_integer():
                result_text = str(int(result))
            else:
                result_text = str(result)
            return {"tool_result": f"Math result: {result_text}"}
        except Exception:
            return {"tool_result": "Math result: unable to parse expression."}

    def code_node(state: AgentState) -> dict[str, str]:
        user_query = state["messages"][-1].lower() if state["messages"] else ""
        if "hello world" in user_query and "python" in user_query:
            return {
                "tool_result": (
                    "```python\n"
                    "def hello_world() -> str:\n"
                    "    return 'Hello, world!'\n"
                    "```"
                )
            }
        return {
            "tool_result": (
                "Code helper: уточни язык и задачу, и я предложу пример реализации."
            )
        }

    def general_node(state: AgentState) -> dict[str, str]:
        _ = state
        return {
            "tool_result": (
                "General response: привет! Могу помочь с вопросами по AI и разработке."
            )
        }

    graph.add_node("classify_node", classify_node)
    graph.add_node("math_node", math_node)
    graph.add_node("code_node", code_node)
    graph.add_node("general_node", general_node)

    graph.add_edge(START, "classify_node")
    graph.add_conditional_edges(
        "classify_node",
        _route_by_classification,
        {
            "math": "math_node",
            "code": "code_node",
            "general": "general_node",
        },
    )
    graph.add_edge("math_node", END)
    graph.add_edge("code_node", END)
    graph.add_edge("general_node", END)

    return graph.compile()


def run_simple_agent(
    user_query: str, classifier_fn: ClassifierFn | None = None
) -> AgentState:
    """Run the compiled graph for a single user request."""
    graph = build_simple_agent_graph(classifier_fn=classifier_fn)
    initial_state: AgentState = {
        "messages": [user_query],
        "classification": "general",
        "tool_result": "",
    }
    return cast(AgentState, graph.invoke(initial_state))
