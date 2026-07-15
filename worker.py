"""ARQ worker for heavy research and web-search tasks."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from arq.connections import RedisSettings

from agents.database import init_db, update_async_task_status
from agents.multi_agent import build_initial_multi_agent_state, build_multi_agent_graph
from agents.web_search import TavilyWebSearch, format_search_results


def _redis_settings_from_env() -> RedisSettings:
    raw = os.getenv("REDIS_URL", "redis://redis:6379").strip()
    parsed = urlparse(raw)
    host = parsed.hostname or "redis"
    port = int(parsed.port or 6379)
    database = int((parsed.path or "/0").strip("/")) if (parsed.path or "").strip("/") else 0
    password = parsed.password
    return RedisSettings(host=host, port=port, database=database, password=password)


async def process_research_task(
    _ctx: dict,
    topic: str,
    user_id: int,
    task_id: str,
) -> str:
    update_async_task_status(task_id, status="running")
    graph = build_multi_agent_graph()
    initial_state = build_initial_multi_agent_state(
        topic=topic,
        user_id=user_id,
        conversation_history=[],
        use_llm=True,
    )
    try:
        result = await graph.ainvoke(initial_state)
        draft = str(result.get("draft", "")).strip()
        if not draft:
            draft = "Пустой результат от research worker."
        update_async_task_status(task_id, status="done", result=draft)
        return draft
    except Exception as exc:
        update_async_task_status(task_id, status="failed", error=str(exc))
        raise


async def process_web_search_task(
    _ctx: dict,
    query: str,
    user_id: int,
    task_id: str,
) -> str:
    del user_id  # reserved for future per-user quotas/rate limits.
    update_async_task_status(task_id, status="running")
    try:
        client = TavilyWebSearch()
        items = await client.search(query, max_results=5)
        if not items:
            text = "Веб-поиск не дал результатов."
        else:
            text = format_search_results(items)
        update_async_task_status(task_id, status="done", result=text)
        return text
    except Exception as exc:
        update_async_task_status(task_id, status="failed", error=str(exc))
        raise


async def health_check(_ctx: dict) -> str:
    return "ok"


async def startup(_ctx: dict) -> None:
    init_db()


class WorkerSettings:
    functions = [process_research_task, process_web_search_task]
    redis_settings = _redis_settings_from_env()
    max_jobs = 10
    job_timeout = 600
    on_startup = startup
    health_check_interval = 60
