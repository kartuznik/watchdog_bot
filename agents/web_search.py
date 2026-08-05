"""Tavily web search adapter for Watchdog Bot."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, TypedDict

from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError

logger = logging.getLogger(__name__)

# Leading interrogatives that poison Tavily ranking toward dictionary pages.
_LEADING_QUESTION_WORDS = frozenset(
    {
        "какие",
        "какой",
        "какая",
        "какое",
        "каким",
        "какими",
        "каких",
        "что",
        "чем",
        "чего",
        "как",
        "кто",
        "кого",
        "кому",
        "где",
        "когда",
        "почему",
        "зачем",
        "откуда",
        "куда",
        "сколько",
        "чей",
        "чья",
        "чьё",
        "чьи",
        "ли",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
    }
)

_STOPWORDS = frozenset(
    {
        "и",
        "в",
        "во",
        "на",
        "по",
        "про",
        "о",
        "об",
        "от",
        "для",
        "из",
        "к",
        "ко",
        "с",
        "со",
        "у",
        "а",
        "но",
        "же",
        "бы",
        "то",
        "это",
        "этот",
        "эта",
        "эти",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "about",
        "is",
        "are",
        "be",
    }
) | _LEADING_QUESTION_WORDS

_MIN_SEARCH_QUERY_CHARS = 8
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)
_SNIPPET_LIMIT = 700


class SourceItem(TypedDict):
    title: str
    url: str
    snippet: str
    published_at: str


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def significant_terms(text: str) -> set[str]:
    """Content-bearing tokens: drop stop/question words and ultra-short noise."""
    terms: set[str] = set()
    for tok in _tokenize(text):
        if tok in _STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        if len(tok) == 2 and not tok.isalpha():
            continue
        terms.add(tok)
    return terms


def build_search_query(original: str, *, min_chars: int = _MIN_SEARCH_QUERY_CHARS) -> str:
    """Soft reformulation for Tavily: strip leading interrogatives/stopwords.

    Gate: if reformulation is too short or loses all significant terms from the
    original, return the full original topic and log a WARNING.
    """
    original_clean = " ".join((original or "").strip().split())
    if not original_clean:
        return ""

    tokens = _tokenize(original_clean)
    while tokens and tokens[0] in _LEADING_QUESTION_WORDS:
        tokens = tokens[1:]

    kept = [t for t in tokens if t not in _STOPWORDS]
    reformulated = " ".join(kept).strip()
    original_sig = significant_terms(original_clean)
    reform_sig = significant_terms(reformulated)

    too_short = len(reformulated) < max(1, int(min_chars))
    lost_signal = bool(original_sig) and not (reform_sig & original_sig)

    if too_short or lost_signal or not reformulated:
        logger.warning(
            "search query gate: falling back to full original "
            "(too_short=%s lost_signal=%s reform=%r original=%r)",
            too_short,
            lost_signal,
            reformulated[:180],
            original_clean[:180],
        )
        return original_clean

    return reformulated


def extract_published_at(item: dict[str, Any]) -> str:
    """Best-effort publication date from Tavily (or similar) result dict."""
    for key in (
        "published_date",
        "published_at",
        "published",
        "date",
        "pub_date",
    ):
        raw = item.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text[:64]
    return ""


def format_evidence_cards(sources: list[SourceItem] | list[dict[str, Any]] | None) -> str:
    """Numbered source cards for research_data / LLM context."""
    items = normalize_source_items(list(sources or []))
    if not items:
        return "Подтверждённые источники отсутствуют."
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        published = str(item.get("published_at") or "").strip() or "дата не указана"
        snippet = str(item.get("snippet") or "").strip() or "(сниппет отсутствует)"
        lines.append(
            f"{idx}. {item['title']}\n"
            f"   URL: {item['url']}\n"
            f"   Дата публикации: {published}\n"
            f"   Сниппет: {snippet}"
        )
    return "\n".join(lines)


def evidence_stats(sources: list[SourceItem] | list[dict[str, Any]] | None) -> tuple[int, int]:
    items = normalize_source_items(list(sources or []))
    chars = sum(len(str(i.get("snippet") or "")) for i in items)
    return len(items), chars


def filter_relevant_sources(
    query: str,
    sources: list[SourceItem] | list[dict[str, Any]] | None,
) -> list[SourceItem]:
    """Drop sources whose title/snippet share no significant terms with the query."""
    query_terms = significant_terms(query)
    normalized = normalize_source_items(list(sources or []))
    if not query_terms:
        return normalized

    kept: list[SourceItem] = []
    for item in normalized:
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        title_terms = significant_terms(f"{title} {snippet}")
        overlap = query_terms & title_terms
        if not overlap:
            logger.info(
                "source relevance filter: drop title=%r url=%r reason=no_term_overlap query_terms=%s",
                title[:120],
                str(item.get("url") or "")[:180],
                sorted(query_terms)[:12],
            )
            continue
        kept.append(item)
    return kept


def normalize_source_items(raw: list[Any] | None) -> list[SourceItem]:
    """Normalize URL strings or dicts into SourceItem list (keeps snippet/date)."""
    items: list[SourceItem] = []
    seen: set[str] = set()
    for entry in raw or []:
        title = ""
        url = ""
        snippet = ""
        published_at = ""
        if isinstance(entry, str):
            url = entry.strip()
            title = url
        elif isinstance(entry, dict):
            url = str(entry.get("url", "")).strip()
            title = str(entry.get("title") or url).strip() or url
            snippet = str(entry.get("snippet") or entry.get("content") or "").strip()
            published_at = str(entry.get("published_at") or entry.get("published_date") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        if len(snippet) > _SNIPPET_LIMIT:
            snippet = snippet[:_SNIPPET_LIMIT].rstrip() + "…"
        items.append(
            {
                "title": title[:120],
                "url": url,
                "snippet": snippet,
                "published_at": published_at[:64],
            }
        )
    return items


class WebSearchTool:
    """Synchronous Tavily search tool for graph nodes and workers."""

    def __init__(self) -> None:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 3) -> str:
        text, _sources = self.search_with_sources(query, max_results=max_results)
        return text

    def search_with_sources(
        self,
        query: str,
        max_results: int = 3,
        *,
        relevance_query: str | None = None,
    ) -> tuple[str, list[SourceItem]]:
        response = self.client.search(query=query, max_results=max_results, timeout=25)
        raw_sources: list[SourceItem] = []
        for item in response.get("results", []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            title = str(item.get("title") or url or "Без названия").strip()
            content = str(item.get("content", "")).strip().replace("\n", " ")
            published_at = extract_published_at(item)
            if not url:
                continue
            if len(content) > _SNIPPET_LIMIT:
                content = content[:_SNIPPET_LIMIT].rstrip() + "…"
            raw_sources.append(
                {
                    "title": title[:120],
                    "url": url,
                    "snippet": content,
                    "published_at": published_at,
                }
            )
        kept = filter_relevant_sources(relevance_query or query, raw_sources)
        return format_evidence_cards(kept), kept


class TavilyWebSearch:
    def __init__(self) -> None:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self._client = TavilyClient(api_key=api_key) if api_key else None

    @property
    def available(self) -> bool:
        return self._client is not None

    async def search(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        if not self._client:
            return []

        def _run() -> list[dict[str, Any]]:
            assert self._client is not None
            result = self._client.search(query=query, max_results=max_results, timeout=25)
            items = result.get("results", [])
            return [item for item in items if isinstance(item, dict)]

        try:
            return await asyncio.to_thread(_run)
        except TavilyTimeoutError:
            logger.warning("Tavily timeout for query=%r", query[:180])
            return []
        except Exception:
            logger.exception("Tavily search failed for query=%r", query[:180])
            return []


def format_search_results(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Веб-поиск не дал результатов или не настроен."

    lines: list[str] = []
    for item in items[:5]:
        title = str(item.get("title", "Без заголовка")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip().replace("\n", " ")
        lines.append(f"- {title}\n  {url}\n  {content[:260]}")
    return "\n".join(lines)
