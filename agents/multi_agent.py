"""Multi-agent LangGraph pipeline: WebSearch -> Researcher -> Writer -> Reviewer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.llm_config import LLMConfig
from agents.metrics import observe_llm_fallback
from agents.tg_parser import extract_telegram_usernames, fetch_many_channels_async
from agents.web_search import WebSearchTool
from config import is_module_enabled

logger = logging.getLogger(__name__)

RouteLabel = Literal["writer_node", "__end__"]
MAX_REVISIONS = 2
APPROVE_SCORE_THRESHOLD = 3.5


class HistoryMessage(TypedDict):
    role: str
    content: str


class MultiAgentState(TypedDict):
    """Shared state for Researcher -> Writer -> Reviewer workflow."""

    user_id: int
    topic: str
    conversation_history: list[HistoryMessage]
    research_data: str
    web_sources: list[str]
    draft: str
    feedback: str
    revision_count: int
    use_llm: bool
    llm_prompt_tokens: int
    llm_completion_tokens: int
    estimated_cost_usd: float


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
    """Invoke primary LLM with automatic OpenAI↔DeepSeek fallback on auth/billing errors."""
    primary = LLMConfig.get_provider()
    providers = [primary]
    alternate = LLMConfig.alternate_provider(primary)
    if alternate != primary:
        providers.append(alternate)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    last_error: BaseException | None = None

    for index, provider in enumerate(providers):
        try:
            llm = LLMConfig.create_chat_model(temperature=temperature, provider=provider)
        except ValueError as exc:
            logger.info("LLM provider %s unavailable: %s", provider, exc)
            last_error = exc
            continue
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            last_error = exc
            if index == 0 and LLMConfig.is_llm_auth_or_balance_error(exc):
                logger.warning(
                    "LLM provider %s failed with auth/billing error (%s); trying %s",
                    provider,
                    type(exc).__name__,
                    alternate,
                )
                continue
            raise
        if index > 0:
            observe_llm_fallback(primary, provider)
            logger.info("LLM fallback succeeded: %s -> %s", primary, provider)
        text = _strip_text(getattr(response, "content", ""))
        prompt_tokens, completion_tokens = _extract_usage(response)
        return text, prompt_tokens, completion_tokens

    if last_error is not None:
        logger.info(
            "All LLM providers failed (%s); using deterministic mode",
            type(last_error).__name__,
        )
    return None


def _history_as_text(history: list[HistoryMessage], limit: int = 8) -> str:
    tail = history[-limit:]
    lines: list[str] = []
    for item in tail:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip() or "История пуста."


def _source_lines(web_sources: list[str], limit: int = 5) -> list[str]:
    return [f"Источник: [{url}]" for url in web_sources[:limit] if str(url).strip()]


def ensure_sources_block(draft: str, web_sources: list[str]) -> str:
    """Attach a mandatory sources block when web search returned URLs."""
    lines = _source_lines(web_sources)
    if not lines:
        return draft
    missing = [url for url in web_sources[:5] if url and url not in draft]
    if not missing and "Источник:" in draft:
        return draft
    block = "\n\n### Источники\n" + "\n".join(lines)
    if "### Источники" in draft:
        return draft
    return draft.rstrip() + block


def _token_cost_update(
    state: MultiAgentState,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, int | float]:
    new_prompt = state["llm_prompt_tokens"] + prompt_tokens
    new_completion = state["llm_completion_tokens"] + completion_tokens
    return {
        "llm_prompt_tokens": new_prompt,
        "llm_completion_tokens": new_completion,
        "estimated_cost_usd": LLMConfig.estimate_cost_usd(new_prompt, new_completion),
    }


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
    if feedback:
        draft = (
            f"Улучшенный материал по теме '{topic}': {research_data}. "
            f"Учтена обратная связь ревьюера: {feedback}. "
            "Добавлены чёткие выводы и структура."
        )
    elif "плохой" in topic.lower():
        draft = (
            f"Черновик по теме '{topic}'. Основа: {research_data}. "
            "Требуется доработка: мало структуры и выводов."
        )
    else:
        draft = (
            f"Готовый материал по теме '{topic}': {research_data}. "
            "Структура ясная, выводы сформулированы."
        )
    return {"draft": ensure_sources_block(draft, state["web_sources"])}


def _mock_review(state: MultiAgentState) -> dict[str, str | int]:
    """Deterministic reviewer without magic words — criteria on draft quality markers."""
    draft = state["draft"].lower()
    needs_work = (
        "требуется доработка" in draft
        or "мало структуры" in draft
        or ("плохой" in state["topic"].lower() and "улучшенный материал" not in draft)
    )
    if needs_work and state["revision_count"] < MAX_REVISIONS:
        return {
            "feedback": (
                "Подними completeness и structure: добавь выводы, "
                "убери размытые формулировки, зафиксируй итог."
            ),
            "revision_count": state["revision_count"] + 1,
        }
    return {"feedback": ""}


async def web_search_node(state: MultiAgentState) -> dict[str, Any]:
    topic = state["topic"]
    if not state["use_llm"] or not is_module_enabled("web_search"):
        return {
            "research_data": "Веб-поиск отключен (feature flag / mock mode).",
            "web_sources": [],
        }

    tavily_text = ""
    tavily_sources: list[str] = []
    try:
        web_tool = WebSearchTool()
        tavily_text, tavily_sources = await asyncio.to_thread(
            web_tool.search_with_sources,
            topic,
            4,
        )
    except ValueError:
        logger.info("Tavily not configured: TAVILY_API_KEY is missing.")
    except Exception:
        logger.exception("Tavily search failed for topic=%r", topic[:180])

    usernames = extract_telegram_usernames(topic)
    tg_posts = await fetch_many_channels_async(usernames, per_channel=2) if usernames else []

    blocks: list[str] = []
    sources: list[str] = list(tavily_sources)
    if tavily_text.strip():
        blocks.append("Веб-результаты (Tavily):\n" + tavily_text.strip())
    else:
        blocks.append("Tavily не настроен (нет TAVILY_API_KEY).")

    if tg_posts:
        lines = []
        for post in tg_posts[:4]:
            title = str(post.get("title", "Пост Telegram")).strip()
            url = str(post.get("url", "")).strip()
            content = str(post.get("content", "")).strip()
            if url:
                sources.append(url)
                lines.append(f"- {title}\n  Источник: [{url}]\n  {content[:220]}")
            else:
                lines.append(f"- {title}\n  {content[:220]}")
        blocks.append("Публичные Telegram-каналы:\n" + "\n".join(lines))

    if not blocks:
        blocks.append("Внешние источники не найдены, используй базовые знания модели.")
    deduplicated_sources = list(dict.fromkeys([s for s in sources if s.strip()]))
    return {
        "research_data": "\n\n".join(blocks),
        "web_sources": deduplicated_sources,
    }


async def research_node(state: MultiAgentState) -> dict[str, str | int | float]:
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
    update = _token_cost_update(state, prompt_tokens, completion_tokens)
    update["research_data"] = text or _mock_research(state)["research_data"]
    return update


async def writer_node(state: MultiAgentState) -> dict[str, str | int | float]:
    """Writer node: draft content from research + reviewer feedback."""
    if not state["use_llm"]:
        return _mock_writer(state)

    feedback = state["feedback"].strip() or "Нет обратной связи."
    source_lines = _source_lines(state["web_sources"])
    sources_context = (
        "Список подтвержденных источников:\n" + "\n".join(source_lines)
        if source_lines
        else "Список подтвержденных источников отсутствует."
    )
    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"Исследование:\n{state['research_data']}\n\n"
        f"{sources_context}\n\n"
        f"Обратная связь ревьюера: {feedback}\n\n"
        "Напиши финальный черновик в 3-5 абзацах. "
        "Если это исправленная версия, явно улучши структуру и ясность. "
        "Если есть источники, обязательно добавь блок с каждой строкой в формате "
        "'Источник: [url]'."
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
    draft = ensure_sources_block(
        text or _mock_writer(state)["draft"],
        state["web_sources"],
    )
    update = _token_cost_update(state, prompt_tokens, completion_tokens)
    update["draft"] = draft
    return update


def parse_reviewer_payload(raw: str) -> tuple[str, dict[str, float], str]:
    """Parse structured reviewer JSON -> (decision, scores, feedback)."""
    text = (raw or "").strip()
    default_scores = {
        "clarity": 3.0,
        "completeness": 3.0,
        "grounding": 3.0,
        "structure": 3.0,
    }
    if not text:
        return "revise", default_scores, "Добавь фактуру, структуру и четкий вывод."

    candidate = text
    if "{" in text and "}" in text:
        candidate = text[text.find("{") : text.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return "revise", default_scores, "Добавь фактуру, структуру и четкий вывод."

    decision_raw = str(parsed.get("decision", "")).strip().lower()
    if decision_raw not in {"approve", "revise"}:
        # Backward-compatible with older {"approved": bool} payloads.
        if "approved" in parsed:
            decision_raw = "approve" if bool(parsed.get("approved")) else "revise"
        else:
            decision_raw = "revise"

    scores_raw = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    scores: dict[str, float] = {}
    for key in ("clarity", "completeness", "grounding", "structure"):
        try:
            value = float(scores_raw.get(key, default_scores[key]))
        except (TypeError, ValueError):
            value = default_scores[key]
        scores[key] = max(1.0, min(5.0, value))

    feedback = str(parsed.get("feedback", "") or "").strip()
    return decision_raw, scores, feedback


def should_approve_review(
    decision: str,
    scores: dict[str, float],
    feedback: str,
    *,
    threshold: float = APPROVE_SCORE_THRESHOLD,
) -> bool:
    avg = sum(scores.values()) / max(1, len(scores))
    if decision == "approve" and avg >= threshold:
        return True
    if decision == "approve" and not feedback.strip() and avg >= threshold - 0.5:
        return True
    return False


async def reviewer_node(state: MultiAgentState) -> dict[str, str | int | float]:
    """Reviewer node: structured JSON scores + approve/revise without magic words."""
    revision_count = state["revision_count"]
    if not state["use_llm"]:
        return _mock_review(state)

    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"Черновик:\n{state['draft']}\n\n"
        "Оцени черновик и верни JSON строго формата:\n"
        "{"
        '"decision": "approve"|"revise", '
        '"scores": {"clarity":1-5, "completeness":1-5, "grounding":1-5, "structure":1-5}, '
        '"feedback": "строка"'
        "}\n"
        "approve только если ответ ясен, полон, опирается на факты и хорошо структурирован. "
        "При revise укажи конкретный feedback."
    )
    result = await _invoke_llm(
        system_prompt=(
            "Ты Reviewer-агент. Оценивай по критериям clarity/completeness/grounding/structure. "
            "Отвечай только валидным JSON без markdown. "
            "Feedback пиши по-русски, лаконично, без воды."
        ),
        user_prompt=user_prompt,
        temperature=0,
    )
    if result is None:
        return _mock_review(state)

    text, prompt_tokens, completion_tokens = result
    decision, scores, feedback = parse_reviewer_payload(text)
    approved = should_approve_review(decision, scores, feedback)
    update = _token_cost_update(state, prompt_tokens, completion_tokens)

    if approved:
        update["feedback"] = ""
        return update

    if revision_count < MAX_REVISIONS:
        update["feedback"] = feedback or (
            "Улучши ясность, полноту и структуру ответа по замечаниям ревьюера."
        )
        update["revision_count"] = revision_count + 1
        return update

    # Hit revision cap — accept current draft.
    update["feedback"] = ""
    return update


def route_after_review(state: MultiAgentState) -> RouteLabel:
    """Route back to writer while revision limit is not reached."""
    has_feedback = bool(state["feedback"].strip())
    if state["revision_count"] < MAX_REVISIONS and has_feedback:
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
        "web_sources": [],
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "use_llm": use_llm,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "estimated_cost_usd": 0.0,
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
