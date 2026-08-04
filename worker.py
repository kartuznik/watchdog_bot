"""ARQ worker for heavy research tasks with stage progress updates."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

from arq.connections import RedisSettings

from agents.database import init_db, update_async_task_stage, update_async_task_status
from agents.metrics import observe_token_usage
from agents.multi_agent import (
    HistoryMessage,
    build_initial_multi_agent_state,
    build_multi_agent_graph,
)
from telegram_bot.progress import stage_for_node


def _redis_settings_from_env() -> RedisSettings:
    raw = os.getenv("REDIS_URL", "redis://redis:6379").strip()
    parsed = urlparse(raw)
    host = parsed.hostname or "redis"
    port = int(parsed.port or 6379)
    database = int((parsed.path or "/0").strip("/")) if (parsed.path or "").strip("/") else 0
    password = parsed.password
    return RedisSettings(host=host, port=port, database=database, password=password)


def _normalize_history(raw: Any) -> list[HistoryMessage]:
    if not isinstance(raw, list):
        return []
    history: list[HistoryMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content})
    return history


async def process_research_task(
    _ctx: dict,
    topic: str,
    user_id: int,
    task_id: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    update_async_task_status(task_id, status="running", stage="routing")
    graph = build_multi_agent_graph()
    history = _normalize_history(conversation_history)
    initial_state = build_initial_multi_agent_state(
        topic=topic,
        user_id=user_id,
        conversation_history=history,
        use_llm=True,
    )
    merged: dict[str, Any] = dict(initial_state)
    try:
        async for event in graph.astream(initial_state, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name, delta in event.items():
                if isinstance(delta, dict):
                    merged.update(delta)
                stage = stage_for_node(str(node_name))
                if stage:
                    update_async_task_stage(task_id, stage)

        draft = str(merged.get("draft", "")).strip()
        if not draft:
            draft = "Пустой результат от research worker."
        observe_token_usage(
            int(merged.get("llm_prompt_tokens", 0) or 0),
            int(merged.get("llm_completion_tokens", 0) or 0),
            cost_usd=float(merged.get("estimated_cost_usd", 0.0) or 0.0),
        )
        payload = {
            "topic": topic,
            "draft": draft,
            "research_summary": str(merged.get("research_summary", "") or "").strip(),
            "research_data": str(merged.get("research_data", "") or "").strip(),
            "web_sources": list(merged.get("web_sources") or []),
            "revision_count": int(merged.get("revision_count", 0) or 0),
            "llm_prompt_tokens": int(merged.get("llm_prompt_tokens", 0) or 0),
            "llm_completion_tokens": int(merged.get("llm_completion_tokens", 0) or 0),
            "estimated_cost_usd": float(merged.get("estimated_cost_usd", 0.0) or 0.0),
            "need_web_search": bool(merged.get("need_web_search", False)),
            "router_mode": str(merged.get("router_mode", "") or ""),
            "router_reason": str(merged.get("router_reason", "") or ""),
        }
        encoded = json.dumps(payload, ensure_ascii=False)
        update_async_task_status(task_id, status="done", result=encoded, stage="done")
        return encoded
    except Exception as exc:
        update_async_task_status(task_id, status="failed", error=str(exc), stage="done")
        raise


async def startup(_ctx: dict) -> None:
    init_db()


class WorkerSettings:
    functions = [process_research_task]
    redis_settings = _redis_settings_from_env()
    max_jobs = 10
    job_timeout = 600
    on_startup = startup
    health_check_interval = 60
