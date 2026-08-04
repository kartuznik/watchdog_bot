"""Tavily web search adapter for Watchdog Bot."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, TypedDict

from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError

logger = logging.getLogger(__name__)


class SourceItem(TypedDict):
    title: str
    url: str


def normalize_source_items(raw: list[Any] | None) -> list[SourceItem]:
    """Normalize URL strings or {title,url} dicts into SourceItem list."""
    items: list[SourceItem] = []
    seen: set[str] = set()
    for entry in raw or []:
        title = ""
        url = ""
        if isinstance(entry, str):
            url = entry.strip()
            title = url
        elif isinstance(entry, dict):
            url = str(entry.get("url", "")).strip()
            title = str(entry.get("title") or url).strip() or url
        if not url or url in seen:
            continue
        seen.add(url)
        items.append({"title": title[:120], "url": url})
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
        self, query: str, max_results: int = 3
    ) -> tuple[str, list[SourceItem]]:
        response = self.client.search(query=query, max_results=max_results, timeout=25)
        lines: list[str] = []
        sources: list[SourceItem] = []
        for item in response.get("results", []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            title = str(item.get("title") or url or "Без названия").strip()
            content = str(item.get("content", "")).strip().replace("\n", " ")
            if url:
                sources.append({"title": title[:120], "url": url})
                lines.append(f"{title}\nURL: {url}\nКонтент: {content}")
            elif content:
                lines.append(f"Контент: {content}")
        return "\n\n".join(lines), normalize_source_items(sources)


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
