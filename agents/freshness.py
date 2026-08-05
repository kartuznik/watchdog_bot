"""Freshness / recency helpers for retrieval and grounded generation."""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

FRESHNESS_MARKERS = (
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

_DATE_PATTERNS = (
    re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b"),
    re.compile(
        r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|"
        r"сентября|октября|ноября|декабря)\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2}),?\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
)

_RU_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

_EN_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_EN_MONTHS_SHORT = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def topic_needs_freshness(topic: str) -> bool:
    text = (topic or "").strip().lower()
    return any(marker in text for marker in FRESHNESS_MARKERS)


def get_tavily_freshness_days() -> int:
    raw = os.getenv("TAVILY_FRESHNESS_DAYS", "2").strip()
    try:
        return max(1, min(30, int(raw)))
    except ValueError:
        return 2


def get_tavily_news_topic() -> str:
    topic = os.getenv("TAVILY_NEWS_TOPIC", "news").strip().lower() or "news"
    if topic not in {"general", "news", "finance"}:
        return "news"
    return topic


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def freshness_start_date(days: int | None = None, *, today: date | None = None) -> str:
    window = get_tavily_freshness_days() if days is None else max(1, int(days))
    day = today or utc_today()
    return (day - timedelta(days=window)).isoformat()


def parse_source_date(
    text: str,
    *,
    today: date | None = None,
    allow_relative: bool = False,
) -> date | None:
    """Best-effort parse of an ISO / RU / EN date from published_at or snippet."""
    blob = (text or "").strip()
    if not blob:
        return None
    ref = today or utc_today()

    m = _DATE_PATTERNS[0].search(blob)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = _DATE_PATTERNS[1].search(blob)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        for day, month in ((d, mo), (mo, d)):
            try:
                return date(y, month, day)
            except ValueError:
                continue

    m = _DATE_PATTERNS[2].search(blob)
    if m:
        day = int(m.group(1))
        month = _RU_MONTHS.get(m.group(2).lower())
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    m = _DATE_PATTERNS[3].search(blob)
    if m:
        month = _EN_MONTHS.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    m = _DATE_PATTERNS[4].search(blob)
    if m:
        day = int(m.group(1))
        month = _EN_MONTHS_SHORT.get(m.group(2).lower()[:3])
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    # Relative markers only when explicitly allowed (never from section titles alone).
    if allow_relative:
        low = blob.lower()
        if re.search(r"\bсегодня\b|\btoday\b", low):
            return ref
        if re.search(r"\bвчера\b|\byesterday\b", low):
            return ref - timedelta(days=1)
    return None


def source_item_date(item: dict[str, Any], *, today: date | None = None) -> date | None:
    published = str(item.get("published_at") or "").strip()
    parsed = parse_source_date(published, today=today, allow_relative=False)
    if parsed:
        return parsed
    snippet = str(item.get("snippet") or "")
    # Prefer absolute dates inside snippet; do not treat "новости сегодня" titles as pubdate.
    parsed = parse_source_date(snippet, today=today, allow_relative=False)
    if parsed:
        return parsed
    title = str(item.get("title") or "")
    return parse_source_date(title, today=today, allow_relative=False)


def source_date_range(
    sources: list[dict[str, Any]] | None,
    *,
    today: date | None = None,
) -> tuple[str, str]:
    dates = [d for d in (source_item_date(s, today=today) for s in (sources or [])) if d]
    if not dates:
        return "", ""
    return min(dates).isoformat(), max(dates).isoformat()


def apply_freshness_ranking(
    sources: list[dict[str, Any]],
    *,
    require_freshness: bool,
    days: int | None = None,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Rank by recency; drop older-than-window when at least one fresh remains.

    Returns (sources, degraded) where degraded=True if we kept stale sources.
    """
    if not require_freshness or not sources:
        return list(sources), False

    window = get_tavily_freshness_days() if days is None else max(1, int(days))
    ref = today or utc_today()
    cutoff = ref - timedelta(days=window)

    decorated: list[tuple[date, dict[str, Any]]] = []
    undated: list[dict[str, Any]] = []
    for item in sources:
        d = source_item_date(item, today=ref)
        if d is None:
            undated.append(item)
        else:
            decorated.append((d, item))

    decorated.sort(key=lambda pair: pair[0], reverse=True)
    fresh = [item for d, item in decorated if d >= cutoff]
    stale = [item for d, item in decorated if d < cutoff]

    if fresh:
        # Prefer dated fresh; append undated at end (unknown recency).
        return fresh + undated, False

    # Degradation: keep best available (newest dated, then undated).
    logger.warning(
        "freshness degradation: no sources within %s days (cutoff=%s); "
        "keeping best available dated=%s undated=%s",
        window,
        cutoff.isoformat(),
        len(stale),
        len(undated),
    )
    return [item for _, item in decorated] + undated, True


def freshness_honesty_note(
    topic: str,
    sources: list[dict[str, Any]] | None,
    *,
    days: int | None = None,
    today: date | None = None,
) -> str:
    """Writer-facing note when freshest source is older than the freshness window."""
    if not topic_needs_freshness(topic):
        return ""
    window = get_tavily_freshness_days() if days is None else max(1, int(days))
    ref = today or utc_today()
    cutoff = ref - timedelta(days=window)
    dates = [d for d in (source_item_date(s, today=ref) for s in (sources or [])) if d]
    if not dates:
        return (
            f"ВНИМАНИЕ: запрос про актуальные новости («сегодня»), но датированных "
            f"источников нет. Не выдавай старые события за новости на {ref.isoformat()}."
        )
    newest = max(dates)
    if newest >= cutoff:
        return ""
    return (
        f"ВНИМАНИЕ: запрос про «сегодня» ({ref.isoformat()}), а самые свежие данные "
        f"в источниках датированы {newest.isoformat()} (старше окна {window} дн.). "
        f"В ответе ЯВНО напиши, что приводишь данные на {newest.isoformat()}, "
        f"а не новости текущего дня. Не маскируй устаревшее под «сегодня»."
    )
