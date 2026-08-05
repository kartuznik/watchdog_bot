"""Multi-agent LangGraph pipeline: WebSearch -> Researcher -> Summary -> Writer -> Reviewer."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.llm_config import LLMConfig
from agents.metrics import observe_llm_fallback, observe_router_decision
from agents.router import merge_router_decision
from agents.tg_parser import extract_telegram_usernames, fetch_many_channels_async
from agents.web_search import (
    SourceItem,
    WebSearchTool,
    build_search_query,
    evidence_stats,
    format_evidence_cards,
    normalize_source_items,
)
from config import is_module_enabled

logger = logging.getLogger(__name__)

RouteLabel = Literal["writer_node", "__end__"]
RouterNext = Literal["web_search_node", "research_node", "writer_node"]
MAX_REVISIONS = 2
APPROVE_SCORE_THRESHOLD = 3.5
RESEARCH_SUMMARY_BUDGET = 800


class HistoryMessage(TypedDict):
    role: str
    content: str


class MultiAgentState(TypedDict):
    """Shared state for Researcher -> Writer -> Reviewer workflow."""

    user_id: int
    topic: str
    conversation_history: list[HistoryMessage]
    research_data: str
    research_summary: str
    web_sources: list[SourceItem]
    source_evidence: list[SourceItem]
    draft: str
    feedback: str
    revision_count: int
    use_llm: bool
    llm_prompt_tokens: int
    llm_completion_tokens: int
    estimated_cost_usd: float
    need_web_search: bool
    router_mode: str
    router_reason: str


_FRESHNESS_MARKERS = (
    "сегодня",
    "сейчас",
    "последн",
    "текущ",
    "свеж",
    "live",
    "breaking",
    "today",
    "latest",
    "current",
)

_GROUNDING_SYSTEM = (
    "ЖЁСТКИЕ ПРАВИЛА ЗАЗЕМЛЕНИЯ:\n"
    "1) Факты и даты бери ТОЛЬКО из блока SOURCE_EVIDENCE (сниппеты и published_at).\n"
    "2) Запрещено использовать parametric memory модели для фактов, событий и дат.\n"
    "3) Если в evidence нет нужных фактов — честно напиши, что именно покрывают источники, "
    "и чего в них нет. Не додумывай.\n"
    "4) Даты указывай только если они есть в published_at или в тексте сниппета."
)


def topic_needs_freshness(topic: str) -> bool:
    text = (topic or "").strip().lower()
    return any(marker in text for marker in _FRESHNESS_MARKERS)


def extract_year_mentions(text: str) -> set[int]:
    years = set()
    for match in re.finditer(r"\b(19|20)\d{2}\b", text or ""):
        years.add(int(match.group(0)))
    return years


def evidence_year_mentions(evidence: list[SourceItem] | list[dict[str, Any]] | None) -> set[int]:
    years: set[int] = set()
    for item in normalize_source_items(list(evidence or [])):
        blob = f"{item.get('published_at', '')} {item.get('snippet', '')} {item.get('title', '')}"
        years |= extract_year_mentions(blob)
    return years


def review_grounding_freshness(
    topic: str,
    draft: str,
    evidence: list[SourceItem] | list[dict[str, Any]] | None,
) -> tuple[bool, str]:
    """Return (needs_revise, feedback) for freshness/grounding mismatches."""
    if not topic_needs_freshness(topic):
        return False, ""
    items = normalize_source_items(list(evidence or []))
    if not items:
        return True, (
            "Запрос требует актуальности, но SOURCE_EVIDENCE пуст. "
            "Перепиши ответ честно: без свежих источников нельзя утверждать новости/даты."
        )

    draft_years = extract_year_mentions(draft)
    evidence_years = evidence_year_mentions(items)
    if draft_years and evidence_years:
        if max(draft_years) < max(evidence_years):
            return True, (
                f"Рассинхрон свежести: черновик опирается на год {max(draft_years)}, "
                f"тогда как источники содержат более свежие датировки "
                f"(до {max(evidence_years)}). Перепиши факты и даты строго по SOURCE_EVIDENCE."
            )

    unsupported = sorted(y for y in draft_years if y not in evidence_years)
    if unsupported and evidence_years:
        return True, (
            "В черновике есть датировки, которых нет в SOURCE_EVIDENCE: "
            f"{', '.join(str(y) for y in unsupported)}. "
            "Убери их или замени датами/фактами из сниппетов."
        )
    return False, ""


def _log_evidence(stage: str, evidence: list[SourceItem] | list[dict[str, Any]] | None) -> None:
    snippets, chars = evidence_stats(evidence)
    logger.info("evidence_%s evidence_snippets=%s evidence_chars=%s", stage, snippets, chars)


def _evidence_block(state: MultiAgentState) -> str:
    evidence = list(state.get("source_evidence") or [])
    if not evidence:
        evidence = list(state.get("web_sources") or [])
    return format_evidence_cards(evidence)


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


def fit_text_budget(text: str, budget: int = RESEARCH_SUMMARY_BUDGET) -> str:
    """Trim text to budget on sentence/word boundaries — never mid-word."""
    content = (text or "").strip()
    if len(content) <= budget:
        return content
    window = content[:budget]
    end = -1
    for match in re.finditer(r"[.!?…](?:\s|$)", window):
        end = match.end()
    if end >= budget // 3:
        return window[:end].rstrip()
    space = window.rfind(" ")
    if space >= budget // 3:
        return window[:space].rstrip() + "…"
    return window.rstrip() + "…"


def _source_context_lines(web_sources: list[SourceItem], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for item in normalize_source_items(web_sources)[:limit]:
        published = str(item.get("published_at") or "").strip()
        suffix = f" ({published})" if published else ""
        lines.append(f"- {item['title']}{suffix}: {item['url']}")
    return lines


def ensure_sources_block(draft: str, web_sources: list[Any]) -> str:
    """Legacy helper kept for compatibility; presentation layer no longer embeds sources."""
    items = normalize_source_items(web_sources)
    if not items:
        return draft
    if any(item["url"] in draft for item in items[:5]) and "Источник" in draft:
        return draft
    lines = [f"Источник: [{item['url']}]" for item in items[:5]]
    return draft.rstrip() + "\n\n### Источники\n" + "\n".join(lines)


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


def _mock_research_summary(state: MultiAgentState) -> dict[str, str]:
    topic = state["topic"]
    summary = (
        f"1. По теме «{topic}» собраны ключевые факты и определения.\n"
        f"2. Отмечены основные риски, ограничения и спорные моменты.\n"
        f"3. Сформулированы практические выводы для читателя.\n"
        f"4. При наличии источников они вынесены отдельным блоком."
    )
    return {"research_summary": fit_text_budget(summary, RESEARCH_SUMMARY_BUDGET)}


def _mock_writer(state: MultiAgentState) -> dict[str, str]:
    topic = state["topic"]
    research_data = state["research_data"]
    evidence = list(state.get("source_evidence") or state.get("web_sources") or [])
    feedback = state["feedback"].strip()
    if evidence and not feedback:
        # Grounded mock: surface first snippet facts/dates into the draft for tests.
        card = evidence[0]
        published = str(card.get("published_at") or "").strip() or "дата не указана"
        snippet = str(card.get("snippet") or "").strip()
        draft = (
            f"По теме «{topic}» согласно источнику «{card.get('title', '')}» "
            f"({published}): {snippet}"
        )
        return {"draft": draft}
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
    return {"draft": draft}


def _mock_review(state: MultiAgentState) -> dict[str, str | int]:
    """Deterministic reviewer — quality markers + grounding/freshness gate."""
    draft = state["draft"]
    needs_revise, freshness_feedback = review_grounding_freshness(
        state["topic"],
        draft,
        list(state.get("source_evidence") or state.get("web_sources") or []),
    )
    if needs_revise and state["revision_count"] < MAX_REVISIONS:
        return {
            "feedback": freshness_feedback,
            "revision_count": state["revision_count"] + 1,
        }

    draft_l = draft.lower()
    needs_work = (
        "требуется доработка" in draft_l
        or "мало структуры" in draft_l
        or ("плохой" in state["topic"].lower() and "улучшенный материал" not in draft_l)
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


async def router_node(state: MultiAgentState) -> dict[str, Any]:
    """Classify whether the query needs live web search."""
    web_enabled = is_module_enabled("web_search")
    llm_raw: str | None = None
    token_update: dict[str, int | float] = {}
    if state["use_llm"]:
        result = await _invoke_llm(
            system_prompt=(
                "Ты Router-агент. Реши, нужен ли живой веб-поиск. "
                "Ответь только JSON: "
                '{"need_web_search": true|false, "mode": "creative"|"factual"|"research", "reason": "..."}. '
                "creative: стихи, шутки, перевод, ролевая игра — поиск не нужен. "
                "factual: определение/объяснение из общих знаний — поиск обычно не нужен. "
                "research: факты, новости, актуальность, сравнения — нужен поиск."
            ),
            user_prompt=f"Запрос пользователя:\n{state['topic']}",
            temperature=0,
        )
        if result is not None:
            llm_raw, prompt_tokens, completion_tokens = result
            token_update = _token_cost_update(state, prompt_tokens, completion_tokens)

    decision = merge_router_decision(
        topic=state["topic"],
        llm_raw=llm_raw,
        web_search_enabled=web_enabled,
    )
    observe_router_decision(str(decision["router_decision"]))
    return {
        "need_web_search": bool(decision["need_web_search"]),
        "router_mode": str(decision["router_mode"]),
        "router_reason": str(decision["router_reason"]),
        **token_update,
    }


def route_after_router(state: MultiAgentState) -> RouterNext:
    if state.get("need_web_search"):
        return "web_search_node"
    mode = str(state.get("router_mode") or "").strip().lower()
    if mode == "creative":
        return "writer_node"
    return "research_node"


async def web_search_node(state: MultiAgentState) -> dict[str, Any]:
    topic = state["topic"]
    if not state["use_llm"] or not is_module_enabled("web_search") or not state.get("need_web_search", True):
        return {
            "research_data": state.get("research_data")
            or "Веб-поиск пропущен роутером или отключён.",
            "web_sources": list(state.get("web_sources") or []),
            "source_evidence": list(state.get("source_evidence") or []),
        }

    tavily_query = build_search_query(topic)
    logger.info(
        "Tavily search original_topic=%r tavily_query=%r",
        topic[:240],
        tavily_query[:240],
    )

    tavily_text = ""
    tavily_sources: list[SourceItem] = []
    try:
        web_tool = WebSearchTool()

        def _run_tavily() -> tuple[str, list[SourceItem]]:
            return web_tool.search_with_sources(
                tavily_query,
                4,
                relevance_query=topic,
            )

        tavily_text, tavily_sources = await asyncio.to_thread(_run_tavily)
    except ValueError:
        logger.info("Tavily not configured: TAVILY_API_KEY is missing.")
    except Exception:
        logger.exception(
            "Tavily search failed original_topic=%r tavily_query=%r",
            topic[:180],
            tavily_query[:180],
        )

    usernames = extract_telegram_usernames(topic)
    tg_posts = await fetch_many_channels_async(usernames, per_channel=2) if usernames else []

    sources: list[SourceItem] = list(tavily_sources)
    if tg_posts:
        for post in tg_posts[:4]:
            title = str(post.get("title", "Пост Telegram")).strip()
            url = str(post.get("url", "")).strip()
            content = str(post.get("content", "")).strip().replace("\n", " ")
            if not url:
                continue
            sources.append(
                {
                    "title": (title or url)[:120],
                    "url": url,
                    "snippet": content[:700],
                    "published_at": "",
                }
            )

    evidence = normalize_source_items(sources)
    cards = format_evidence_cards(evidence)
    _log_evidence("web_search", evidence)

    if evidence:
        research_data = "SOURCE_EVIDENCE (immutable cards):\n" + cards
    elif tavily_text.strip():
        research_data = tavily_text.strip()
    else:
        research_data = (
            "Внешние источники не найдены. Нельзя утверждать свежие факты без evidence."
        )

    return {
        "research_data": research_data,
        "web_sources": evidence,
        "source_evidence": evidence,
    }


async def research_node(state: MultiAgentState) -> dict[str, str | int | float]:
    """Researcher node: grounded synthesis over immutable source_evidence."""
    evidence = list(state.get("source_evidence") or state.get("web_sources") or [])
    _log_evidence("research", evidence)
    if not state["use_llm"]:
        if evidence:
            return {
                "research_data": (
                    "Grounded synthesis:\n"
                    + format_evidence_cards(evidence)
                )
            }
        return _mock_research(state)

    history_text = _history_as_text(state["conversation_history"])
    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"История диалога:\n{history_text}\n\n"
        f"SOURCE_EVIDENCE:\n{_evidence_block(state)}\n\n"
        "Собери краткое заземлённое исследование: только факты/даты из SOURCE_EVIDENCE. "
        "Если данных мало — явно перечисли пробелы покрытия источников."
    )
    result = await _invoke_llm(
        system_prompt=(
            "Ты Researcher-агент. Пиши на русском. "
            f"{_GROUNDING_SYSTEM}"
        ),
        user_prompt=user_prompt,
        temperature=0.1,
    )
    if result is None:
        return _mock_research(state)
    text, prompt_tokens, completion_tokens = result
    update = _token_cost_update(state, prompt_tokens, completion_tokens)
    # Keep evidence cards visible for downstream nodes inside research_data synthesis.
    synthesis = (text or "").strip() or _mock_research(state)["research_data"]
    update["research_data"] = (
        "SOURCE_EVIDENCE (immutable cards):\n"
        f"{_evidence_block(state)}\n\n"
        f"GROUNDED_SYNTHESIS:\n{synthesis}"
    )
    return update


async def research_summary_node(state: MultiAgentState) -> dict[str, str | int | float]:
    """Compact 3–5 bullet summary grounded in source_evidence / research_data."""
    if not state["use_llm"]:
        evidence = list(state.get("source_evidence") or [])
        if evidence:
            lines = []
            for idx, item in enumerate(evidence[:4], start=1):
                published = item.get("published_at") or "дата не указана"
                snippet = (item.get("snippet") or "")[:160]
                lines.append(f"{idx}. ({published}) {snippet}")
            summary = "\n".join(lines) or _mock_research_summary(state)["research_summary"]
            return {"research_summary": fit_text_budget(summary, RESEARCH_SUMMARY_BUDGET)}
        return _mock_research_summary(state)

    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"SOURCE_EVIDENCE:\n{_evidence_block(state)}\n\n"
        f"GROUNDED_SYNTHESIS:\n{state['research_data']}\n\n"
        "Сделай компактное саммари из 3-5 нумерованных пунктов на русском "
        "ТОЛЬКО по evidence/synthesis выше. "
        f"Жёсткий бюджет: не больше {RESEARCH_SUMMARY_BUDGET} символов. "
        "Не обрывай слова. Без markdown-заголовков и без ссылок. "
        "Если факт/дата не в evidence — не пиши его."
    )
    result = await _invoke_llm(
        system_prompt=(
            "Ты редактор-суммаризатор. Пиши только нумерованный список 3-5 пунктов. "
            f"{_GROUNDING_SYSTEM}"
        ),
        user_prompt=user_prompt,
        temperature=0,
    )
    if result is None:
        return _mock_research_summary(state)
    text, prompt_tokens, completion_tokens = result
    summary = fit_text_budget(text or "", RESEARCH_SUMMARY_BUDGET)
    if not summary:
        return {
            **_mock_research_summary(state),
            **_token_cost_update(state, prompt_tokens, completion_tokens),
        }
    update = _token_cost_update(state, prompt_tokens, completion_tokens)
    update["research_summary"] = summary
    return update


async def writer_node(state: MultiAgentState) -> dict[str, str | int | float]:
    """Writer node: draft content grounded in source_evidence + synthesis."""
    evidence = list(state.get("source_evidence") or state.get("web_sources") or [])
    _log_evidence("writer", evidence)
    if not state["use_llm"]:
        return _mock_writer(state)

    feedback = state["feedback"].strip() or "Нет обратной связи."
    is_creative = str(state.get("router_mode") or "") == "creative" and not evidence
    if is_creative:
        user_prompt = (
            f"Тема: {state['topic']}\n\n"
            f"Обратная связь ревьюера: {feedback}\n\n"
            "Креативный ответ без претензии на свежие факты/новости. "
            "НЕ добавляй список источников и НЕ используй markdown-заголовки с #."
        )
        system_prompt = (
            "Ты Writer-агент. Пиши ясно, с юмором. "
            "Это креативный режим — не выдумывай новостные факты и даты."
        )
    else:
        user_prompt = (
            f"Тема: {state['topic']}\n\n"
            f"SOURCE_EVIDENCE:\n{_evidence_block(state)}\n\n"
            f"GROUNDED_SYNTHESIS:\n{state['research_data']}\n\n"
            f"Обратная связь ревьюера: {feedback}\n\n"
            "Напиши финальный ответ в 3-5 коротких абзацах. "
            "Каждый факт и каждая дата — только из SOURCE_EVIDENCE. "
            "Если evidence не покрывает вопрос — скажи об этом прямо. "
            "НЕ добавляй список источников и НЕ используй markdown-заголовки с # — "
            "источники уйдут отдельным сообщением."
        )
        system_prompt = (
            "Ты Writer-агент. Пиши ясно, структурно, практично. "
            f"{_GROUNDING_SYSTEM} "
            "Отвечай по-русски, лаконично. БЕЗ воды и канцеляризмов."
        )
    result = await _invoke_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
    )
    if result is None:
        return _mock_writer(state)
    text, prompt_tokens, completion_tokens = result
    draft = (text or _mock_writer(state)["draft"]).strip()
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
    """Reviewer node: grounding/freshness gate + structured JSON scores."""
    revision_count = state["revision_count"]
    evidence = list(state.get("source_evidence") or state.get("web_sources") or [])
    needs_revise, freshness_feedback = review_grounding_freshness(
        state["topic"],
        state["draft"],
        evidence,
    )
    if needs_revise and revision_count < MAX_REVISIONS:
        logger.info("reviewer freshness gate revise: %s", freshness_feedback[:200])
        update: dict[str, str | int | float] = {
            "feedback": freshness_feedback,
            "revision_count": revision_count + 1,
        }
        return update

    if not state["use_llm"]:
        return _mock_review(state)

    freshness_note = (
        "Запрос содержит маркеры актуальности: grounding и свежесть обязательны. "
        "Любая дата/факт вне SOURCE_EVIDENCE → decision=revise."
        if topic_needs_freshness(state["topic"])
        else "Проверь grounding относительно SOURCE_EVIDENCE."
    )
    user_prompt = (
        f"Тема: {state['topic']}\n\n"
        f"SOURCE_EVIDENCE:\n{_evidence_block(state)}\n\n"
        f"Черновик:\n{state['draft']}\n\n"
        f"{freshness_note}\n"
        "Оцени черновик и верни JSON строго формата:\n"
        "{"
        '"decision": "approve"|"revise", '
        '"scores": {"clarity":1-5, "completeness":1-5, "grounding":1-5, "structure":1-5}, '
        '"feedback": "строка"'
        "}\n"
        "approve только если ответ ясен, полон, заземлён в SOURCE_EVIDENCE и хорошо структурирован. "
        "При revise укажи конкретный feedback о рассинхроне фактов/дат."
    )
    result = await _invoke_llm(
        system_prompt=(
            "Ты Reviewer-агент. Оценивай clarity/completeness/grounding/structure. "
            f"{_GROUNDING_SYSTEM} "
            "Отвечай только валидным JSON без markdown. "
            "Feedback пиши по-русски, лаконично."
        ),
        user_prompt=user_prompt,
        temperature=0,
    )
    if result is None:
        return _mock_review(state)

    text, prompt_tokens, completion_tokens = result
    decision, scores, feedback = parse_reviewer_payload(text)
    # Hard floor on grounding for freshness queries.
    if topic_needs_freshness(state["topic"]) and scores.get("grounding", 5) < 4.0:
        decision = "revise"
        if not feedback.strip():
            feedback = (
                "Низкий grounding при актуальном запросе: перепиши факты/даты "
                "строго по SOURCE_EVIDENCE."
            )
    approved = should_approve_review(decision, scores, feedback)
    update = _token_cost_update(state, prompt_tokens, completion_tokens)

    if approved:
        update["feedback"] = ""
        return update

    if revision_count < MAX_REVISIONS:
        update["feedback"] = feedback or (
            "Улучши ясность, полноту и grounding ответа по SOURCE_EVIDENCE."
        )
        update["revision_count"] = revision_count + 1
        return update

    update["feedback"] = ""
    return update


def route_after_review(state: MultiAgentState) -> RouteLabel:
    """Route back to writer while revision limit is not reached."""
    has_feedback = bool(state["feedback"].strip())
    if state["revision_count"] < MAX_REVISIONS and has_feedback:
        return "writer_node"
    return "__end__"


def build_multi_agent_graph():
    """Build Router -> (WebSearch?) -> Researcher? -> Summary? -> Writer -> Reviewer graph."""
    graph = StateGraph(MultiAgentState)

    graph.add_node("router_node", router_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("research_node", research_node)
    graph.add_node("research_summary_node", research_summary_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("reviewer_node", reviewer_node)

    graph.add_edge(START, "router_node")
    graph.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "web_search_node": "web_search_node",
            "research_node": "research_node",
            "writer_node": "writer_node",
        },
    )
    graph.add_edge("web_search_node", "research_node")
    graph.add_edge("research_node", "research_summary_node")
    graph.add_edge("research_summary_node", "writer_node")
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
        "research_summary": "",
        "web_sources": [],
        "source_evidence": [],
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "use_llm": use_llm,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "estimated_cost_usd": 0.0,
        "need_web_search": False,
        "router_mode": "factual",
        "router_reason": "",
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
