"""Query router: decide whether web search is needed before the multi-agent pipeline."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

logger = logging.getLogger(__name__)

RouterMode = Literal["creative", "factual", "research"]

_SEARCH_MARKERS = (
    "актуальн",
    "новост",
    "сегодня",
    "сейчас",
    "2024",
    "2025",
    "2026",
    "курс ",
    "цена",
    "рынок",
    "сравни",
    "тенденц",
    "последн",
    "кто победил",
    "свеж",
    "live",
    "breaking",
    "по данным",
    "статистик",
)

_CREATIVE_MARKERS = (
    "стих",
    "стихотворен",
    "напиши песн",
    "сочини",
    "сказк",
    "анекдот",
    "шутк",
    "переведи",
    "перефразируй",
    "ролевая",
    "как будто ты",
)

_KNOWLEDGE_MARKERS = (
    "что такое",
    "что значит",
    "объясни термин",
    "объясни понятие",
    "определение",
    "определи",
    "означает",
)


def heuristic_route(topic: str) -> tuple[bool, RouterMode, str]:
    """
    Case-insensitive keyword router.
    Returns (need_web_search, mode, reason).
    """
    text = (topic or "").strip().lower()
    if not text:
        return False, "creative", "empty_topic"

    if any(marker in text for marker in _CREATIVE_MARKERS):
        return False, "creative", "creative_heuristic"

    if any(marker in text for marker in _SEARCH_MARKERS):
        return True, "research", "search_heuristic"

    if any(marker in text for marker in _KNOWLEDGE_MARKERS):
        return False, "factual", "knowledge_heuristic"

    # Unknown factual/research topics: prefer search (sources help grounding).
    if len(text) >= 24 or bool(re.search(r"\b(как|почему|история|доказа|теори)\w*", text)):
        return True, "research", "default_prefer_search"

    return False, "factual", "default_short_factual"


def parse_router_payload(raw: str) -> tuple[bool, RouterMode, str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    candidate = text
    if "{" in text and "}" in text:
        candidate = text[text.find("{") : text.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    need = parsed.get("need_web_search")
    if need is None and "need_search" in parsed:
        need = parsed.get("need_search")
    if not isinstance(need, bool):
        return None

    mode_raw = str(parsed.get("mode", "research")).strip().lower()
    mode: RouterMode
    if mode_raw in {"creative", "factual", "research"}:
        mode = mode_raw  # type: ignore[assignment]
    elif need:
        mode = "research"
    else:
        mode = "factual"

    reason = str(parsed.get("reason", "") or "llm_router").strip() or "llm_router"
    return need, mode, reason


def merge_router_decision(
    *,
    topic: str,
    llm_raw: str | None,
    web_search_enabled: bool,
) -> dict[str, Any]:
    """Combine LLM JSON (if any) with heuristic fallback and feature flag."""
    heuristic = heuristic_route(topic)
    parsed = parse_router_payload(llm_raw or "") if llm_raw else None
    if parsed is None:
        need, mode, reason = heuristic
        source = "heuristic"
    else:
        need, mode, reason = parsed
        source = "llm"

    if not web_search_enabled:
        need = False
        reason = f"{reason}|web_search_disabled"
        if mode == "research":
            mode = "factual"

    decision = "search" if need else "no_search"
    logger.info(
        "router decision=%s mode=%s source=%s reason=%s topic=%r",
        decision,
        mode,
        source,
        reason,
        (topic or "")[:160],
    )
    return {
        "need_web_search": need,
        "router_mode": mode,
        "router_reason": f"{source}:{reason}",
        "router_decision": decision,
    }
