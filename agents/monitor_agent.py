"""LangGraph monitor agent for periodic smart diagnostics."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, TypedDict, cast

from aiogram import Bot
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.database import count_error_responses, count_user_requests, count_users
from agents.health_check import HealthSnapshot
from agents.llm_config import LLMConfig

logger = logging.getLogger(__name__)

Decision = Literal["restart", "notify", "ignore"]


class MonitorAgentState(TypedDict):
    bot_health: dict[str, Any]
    active_tasks: int
    error_rate: float
    memory_usage: float
    analysis: str
    decision: Decision
    alert_text: str
    admin_user_id: int


async def gather_metrics_node(state: MonitorAgentState) -> dict[str, Any]:
    total_requests = max(1, count_user_requests())
    errors = count_error_responses()
    users = count_users()
    error_rate = errors / total_requests
    bot_health = dict(state.get("bot_health", {}))
    bot_health.setdefault("users_count", users)
    bot_health.setdefault("errors_count", errors)
    bot_health.setdefault("requests_count", total_requests)
    return {
        "active_tasks": len(asyncio.all_tasks()),
        "error_rate": error_rate,
        "bot_health": bot_health,
    }


async def analyze_node(state: MonitorAgentState) -> dict[str, str]:
    summary = (
        f"bot_health={json.dumps(state['bot_health'], ensure_ascii=False)}\n"
        f"active_tasks={state['active_tasks']}\n"
        f"error_rate={state['error_rate']:.4f}\n"
        f"memory_usage={state['memory_usage']:.2f}"
    )
    try:
        llm = LLMConfig.create_chat_model(temperature=0)
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Ты monitor-агент для production Telegram-бота. "
                        "Анализируй health-метрики и верни JSON: "
                        '{"action":"restart|notify|ignore","reason":"..."}'
                    )
                ),
                HumanMessage(content=summary),
            ]
        )
        text = str(getattr(response, "content", "")).strip()
        if text:
            return {"analysis": text}
    except Exception:
        logger.exception("Monitor agent LLM analyze failed, fallback to heuristic.")

    # Heuristic fallback when LLM unavailable.
    if state["memory_usage"] > 95:
        return {"analysis": '{"action":"restart","reason":"memory usage is critical"}'}
    if state["error_rate"] > 0.2:
        return {"analysis": '{"action":"notify","reason":"error rate is above threshold"}'}
    if not bool(state["bot_health"].get("telegram_ok", True)):
        return {"analysis": '{"action":"notify","reason":"telegram check failed"}'}
    return {"analysis": '{"action":"ignore","reason":"system is healthy"}'}


async def decide_action_node(state: MonitorAgentState) -> dict[str, str]:
    raw = state.get("analysis", "").strip()
    action: Decision = "ignore"
    reason = "No actions required."
    try:
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw else raw)
        candidate = str(data.get("action", "ignore")).strip().lower()
        if candidate in {"restart", "notify", "ignore"}:
            action = cast(Decision, candidate)
        reason = str(data.get("reason", reason)).strip() or reason
    except Exception:
        lowered = raw.lower()
        if "restart" in lowered:
            action = "restart"
        elif "notify" in lowered or "alert" in lowered:
            action = "notify"
        reason = raw or reason
    return {
        "decision": action,
        "alert_text": f"[MonitorAgent] action={action}; reason={reason}",
    }


async def send_alert_node(state: MonitorAgentState) -> dict[str, str]:
    # Delivery happens in runner where bot instance is available.
    return {"alert_text": state["alert_text"]}


def build_monitor_graph():
    graph = StateGraph(MonitorAgentState)
    graph.add_node("gather_metrics_node", gather_metrics_node)
    graph.add_node("analyze_node", analyze_node)
    graph.add_node("decide_action_node", decide_action_node)
    graph.add_node("send_alert_node", send_alert_node)
    graph.add_edge(START, "gather_metrics_node")
    graph.add_edge("gather_metrics_node", "analyze_node")
    graph.add_edge("analyze_node", "decide_action_node")
    graph.add_edge("decide_action_node", "send_alert_node")
    graph.add_edge("send_alert_node", END)
    return graph.compile()


async def run_monitor_agent(
    *,
    bot: Bot,
    admin_user_id: int | None,
    health_snapshot: HealthSnapshot,
) -> MonitorAgentState:
    graph = build_monitor_graph()
    initial_state: MonitorAgentState = {
        "bot_health": {
            "db_ok": health_snapshot.db_ok,
            "telegram_ok": health_snapshot.telegram_ok,
            "health_message": health_snapshot.message,
        },
        "active_tasks": 0,
        "error_rate": 0.0,
        "memory_usage": health_snapshot.memory_usage,
        "analysis": "",
        "decision": "ignore",
        "alert_text": "",
        "admin_user_id": int(admin_user_id or 0),
    }
    result = cast(MonitorAgentState, await graph.ainvoke(initial_state))

    if result["decision"] in {"notify", "restart"} and admin_user_id:
        try:
            await bot.send_message(chat_id=admin_user_id, text=result["alert_text"])
        except Exception:
            logger.exception("Failed to send monitor alert to admin")
    return result
