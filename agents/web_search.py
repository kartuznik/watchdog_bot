"""Tavily web search adapter inspired by telegram-ai-bot search_with_tavily."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError

logger = logging.getLogger(__name__)


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
