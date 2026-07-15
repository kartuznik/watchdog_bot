"""Production-oriented multi-agent graph with LLM, search and review loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.llm_config import LLMConfig
from agents.tg_parser import extract_telegram_usernames, fetch_many_channels_async
from agents.web_search import TavilyWebSearch, format_search_results

logger = logging.getLogger(__name__)

RouteLabel = Literal["writer_node", "__end__"]


class HistoryMessage(TypedDict):
    role: str
    content: str


class MultiAgentState(TypedDict):
    """Shared state for Researcher -> Writer -> Reviewer workflow."""

    user_id: int
    topic: str
    conversation_history: list[HistoryMessage]
    research_data: str
    draft: str
    feedback: str
    revision_count: int
    use_llm: bool
    llm_prompt_tokens: int
    llm_completion_tokens: int


def _extract_usage(message: Any) -> tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens = int(
            usage_metadata.get("input_tokens")
            or usage_metadata.get("prompt_tokens")
            or 0
        )
        completion_tokens = int(
            usage_metadata.get("output_tokens")
            or usage_metadata.get("completion_tokens")
            or 0
        )
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            prompt_tokens = int(token_usage.get("prompt_tokens", prompt_tokens))
            completion_tokens = int(
                token_usage.get("completion_tokens", completion_tokens)
            )
    return max(0, prompt_tokens), max(0, completion_tokens)


def _strip_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return str(content or "").strip()


async def _invoke_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> tuple[str, int, int] | None:
    try:
        llm = LLMConfig.create_chat_model(temperature=temperature)
    except ValueError:
        logger.info("LLM unavailable, fallback to deterministic mode")
        return None
    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    text = _strip_text(getattr(response, "content", ""))
    prompt_tokens, completion_tokens = _extract_usage(response)
    return text, prompt_tokens, completion_tokens


def _history_as_text(history: list[HistoryMessage], limit: int = 8) -> str:
    tail = history[-limit:]
    lines: list[str] = []
    for item in tail:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip() or "История пуста."


def _mock_research(state: MultiAgentState) -> dict[str, str]:
    topic = state["topic"]
    return {
        "research_data": (
            f"Исследование по теме '{topic}': ключевые идеи, риски, примеры и рекомендации."
        )
    }


def _mock_writer(state: MultiAgentState) -> dict[str, str]:
    topic = state["topic"]
    research_data = state["research_data"]
    feedback = state["feedback"].strip()
    if not feedback and "плохой" in topic.lower():
        return {
            "draft": (
                f"Черновик по теме '{topic}'. Основа: {research_data}. Требуется доработка."
            )
        }
    if feedback:
        return {
            "draft": (
                f"Улучшенный материал по теме '{topic}': {research_data}. "
                f"Учтена обратная связь: {feedback}. Итог отлично."
            )
        }
    return {
        "draft": (
            f"Черновик по теме '{topic}': {research_data}. Качество: отлично."
        )
    }


def _mock_review(state: MultiAgentState) -> dict[str, str | int]:
    if "отлично" not in state["draft"].lower():
        return {
            "feedback": "Добавь четкие выводы и сформулируй финальную версию лучше.",
            "revision_count": state["revision_count"] + 1,
        }
    return {"feedback": ""}


async def web_search_node(state: MultiAgentState) -> dict[str, str]:
    topic = state["topic"]
    if not state["use_llm"]:
        return {"research_data": "Веб-поиск отключен (test/mock mode)."}

    search_client = TavilyWebSearch()
    tavily_results = await search_client.search(topic, max_results=4)
    usernames = extract_telegram_usernames(topic)
    tg_posts = await fetch_many_channels_async(usernames, per_channel=2) if usernames else []

    blocks: list[str] = []
    if tavily_results:
        blocks.append("Веб-результаты (Tavily):\n" + format_search_results(tavily_results))
    elif not search_client.available:
        blocks.append("Tavily не настроен (нет TAVILY_API_KEY).")

    if tg_posts:
        lines = []
        for post in tg_posts[:4]:
            title = str(post.get("title", "Пост Telegram")).strip()
            url = str(post.get("url", "")).strip()
            content = str(post.get("content", "")).strip()
            lines.append(f"- {title}\n  {url}\n  {content[:220]}")
        blocks.append("Публичные Telegram-каналы:\n" + "\n".join(lines))

    if not blocks:
        blocks.append("Внешние источники не найдены, используй базовые знания модели.")
    return {"research_data": "\n\n".join(blocks)}


async def research_node(state: MultiAgentState) -> dict[str, str | int]:
    """Researcher node: collect and synthesize context."""
    if not state["use_llm"]:
        return _mock_research(state)

    history_text = _history_as_text(state["conversation_history"])
    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"История диалога:\n{history_text}\n\n"
        f"Черновые внешние данные:\n{state['research_data']}\n\n"
        "Собери краткое исследование: ключевые факты, риски, практические выводы."
    )
    result = await _invoke_llm(
        system_prompt=(
            "Ты Researcher-агент. Пиши на русском. "
            "Дай плотное, факт-ориентированное исследование без воды."
        ),
        user_prompt=user_prompt,
        temperature=0.1,
    )
    if result is None:
        return _mock_research(state)
    text, prompt_tokens, completion_tokens = result
    return {
        "research_data": text or _mock_research(state)["research_data"],
        "llm_prompt_tokens": state["llm_prompt_tokens"] + prompt_tokens,
        "llm_completion_tokens": state["llm_completion_tokens"] + completion_tokens,
    }


async def writer_node(state: MultiAgentState) -> dict[str, str | int]:
    """Writer node: draft content from research + reviewer feedback."""
    if not state["use_llm"]:
        return _mock_writer(state)

    feedback = state["feedback"].strip() or "Нет обратной связи."
    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"Исследование:\n{state['research_data']}\n\n"
        f"Обратная связь ревьюера: {feedback}\n\n"
        "Напиши финальный черновик в 3-5 абзацах. "
        "Если это исправленная версия, явно улучши структуру и ясность."
    )
    result = await _invoke_llm(
        system_prompt=(
            "Ты Writer-агент. Пиши ясно, структурно, практично. "
            "Не выдумывай факты, опирайся на исследование. "
            "Отвечай по-русски, лаконично, с юмором и метафорами. "
            "БЕЗ воды и канцеляризмов. Максимум 3-5 предложений. "
            "Если можешь ответить кратко — отвечай кратко."
        ),
        user_prompt=user_prompt,
        temperature=0.2,
    )
    if result is None:
        return _mock_writer(state)
    text, prompt_tokens, completion_tokens = result
    draft = text or _mock_writer(state)["draft"]
    if state["feedback"].strip() and "отлично" not in draft.lower():
        draft = draft.rstrip() + "\n\nИтог: отлично."
    return {
        "draft": draft,
        "llm_prompt_tokens": state["llm_prompt_tokens"] + prompt_tokens,
        "llm_completion_tokens": state["llm_completion_tokens"] + completion_tokens,
    }


def _parse_reviewer_json(raw: str) -> tuple[bool, str]:
    text = (raw or "").strip()
    if not text:
        return False, "Добавь фактуру, структуру и четкий вывод."
    candidate = text
    if "{" in text and "}" in text:
        candidate = text[text.find("{") : text.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return False, "Добавь фактуру, структуру и четкий вывод."
    approved = bool(parsed.get("approved", False))
    feedback = str(parsed.get("feedback", "") or "").strip()
    return approved, feedback


async def reviewer_node(state: MultiAgentState) -> dict[str, str | int]:
    """Reviewer node: decide approve/revise with loop limit."""
    revision_count = state["revision_count"]
    if not state["use_llm"]:
        return _mock_review(state)

    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"Черновик:\n{state['draft']}\n\n"
        "Верни JSON строго формата: "
        '{"approved": true|false, "feedback": "строка"}'
    )
    result = await _invoke_llm(
        system_prompt=(
            "Ты Reviewer-агент. Проверяй ясность, полноту и соответствие теме. "
            "Если всё хорошо — approved=true, feedback=''. "
            "Формулируй feedback по-русски, лаконично, с юмором и метафорами, "
            "без воды и канцеляризмов, максимум 3-5 предложений."
        ),
        user_prompt=user_prompt,
        temperature=0,
    )
    if result is None:
        return _mock_review(state)
    text, prompt_tokens, completion_tokens = result
    approved, feedback = _parse_reviewer_json(text)
    if approved or not feedback.strip():
        return {
            "feedback": "",
            "llm_prompt_tokens": state["llm_prompt_tokens"] + prompt_tokens,
            "llm_completion_tokens": state["llm_completion_tokens"] + completion_tokens,
        }
    if revision_count < 2:
        return {
            "feedback": feedback,
            "revision_count": revision_count + 1,
            "llm_prompt_tokens": state["llm_prompt_tokens"] + prompt_tokens,
            "llm_completion_tokens": state["llm_completion_tokens"] + completion_tokens,
        }
    return {
        "feedback": "",
        "llm_prompt_tokens": state["llm_prompt_tokens"] + prompt_tokens,
        "llm_completion_tokens": state["llm_completion_tokens"] + completion_tokens,
    }


def route_after_review(state: MultiAgentState) -> RouteLabel:
    """Route back to writer while revision limit is not reached."""
    has_feedback = bool(state["feedback"].strip())
    if state["revision_count"] < 2 and has_feedback:
        return "writer_node"
    return "__end__"


def build_multi_agent_graph():
    """Build and compile WebSearch -> Researcher -> Writer -> Reviewer graph."""
    graph = StateGraph(MultiAgentState)

    graph.add_node("web_search_node", web_search_node)
    graph.add_node("research_node", research_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("reviewer_node", reviewer_node)

    graph.add_edge(START, "web_search_node")
    graph.add_edge("web_search_node", "research_node")
    graph.add_edge("research_node", "writer_node")
    graph.add_edge("writer_node", "reviewer_node")
    graph.add_conditional_edges(
        "reviewer_node",
        route_after_review,
        {"writer_node": "writer_node", "__end__": END},
    )
    return graph.compile()


def build_initial_multi_agent_state(
    topic: str,
    user_id: int,
    *,
    conversation_history: list[HistoryMessage] | None = None,
    use_llm: bool = True,
) -> MultiAgentState:
    """Create a fresh state for one user/topic run."""
    return {
        "user_id": user_id,
        "topic": topic,
        "conversation_history": list(conversation_history or []),
        "research_data": "",
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "use_llm": use_llm,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
    }


def run_multi_agent(topic: str, user_id: int = 0, use_llm: bool = False) -> MultiAgentState:
    """Run workflow with initial state."""
    graph = build_multi_agent_graph()
    initial_state = build_initial_multi_agent_state(
        topic=topic,
        user_id=user_id,
        use_llm=use_llm,
    )
    return cast(MultiAgentState, asyncio.run(graph.ainvoke(initial_state)))
