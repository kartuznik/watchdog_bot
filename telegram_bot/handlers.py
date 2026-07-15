"""Message handlers for Telegram + LangGraph integration."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import suppress
from typing import cast

from aiogram import Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

from agents.database import count_conversations, count_users
from agents.health_check import HealthCheckService
from agents.llm_config import LLMConfig
from agents.monitor_agent import MonitorAgentState, run_monitor_agent
from agents.multi_agent import (
    MultiAgentState,
    build_initial_multi_agent_state,
    build_multi_agent_graph,
)
from agents.memory import ChatMemory
from agents.metrics import (
    agent_request_duration_seconds,
    agent_requests_failed_total,
    agent_requests_total,
    observe_token_usage,
)
from agents.roles import get_role, list_admins, remove_role, set_role
from telegram_bot.middlewares.role_check import require_role

logger = logging.getLogger(__name__)
router = Router()
multi_agent_graph = build_multi_agent_graph()
chat_memory = ChatMemory(max_messages=20)
PROCESS_STARTED_AT = time.time()
runtime_health_checker: HealthCheckService | None = None
runtime_last_monitor_state: MonitorAgentState | None = None


def configure_runtime_services(*, health_checker: HealthCheckService | None) -> None:
    global runtime_health_checker
    runtime_health_checker = health_checker


def set_last_monitor_state(state: MonitorAgentState) -> None:
    global runtime_last_monitor_state
    runtime_last_monitor_state = state


def _format_result_markdown(result: MultiAgentState) -> str:
    topic = result["topic"]
    research_data = result["research_data"]
    draft = result["draft"]
    return (
        "## ✅ Готово\n"
        f"**Тема:** {topic}\n\n"
        "### 🔬 Research\n"
        f"{research_data}\n\n"
        "### 📝 Draft\n"
        f"{draft}\n\n"
        f"_Итераций ревью: {result['revision_count']}_"
    )


async def _run_research_flow(message: Message, topic: str) -> None:
    if not topic.strip():
        await message.answer(
            "Укажи тему после команды.\nПример: `/research агенты в поддержке клиентов`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    user_id = message.from_user.id if message.from_user else 0
    conversation_history = chat_memory.get_user_memory(user_id)
    initial_state = build_initial_multi_agent_state(
        topic=topic.strip(),
        user_id=user_id,
        conversation_history=conversation_history,
        use_llm=True,
    )

    typing_task = asyncio.create_task(_typing_pulse(message))
    started_at = time.perf_counter()

    try:
        result = cast(
            MultiAgentState,
            await multi_agent_graph.ainvoke(initial_state),
        )
        elapsed = time.perf_counter() - started_at
        agent_requests_total.inc()
        agent_request_duration_seconds.observe(elapsed)
        observe_token_usage(
            result.get("llm_prompt_tokens", 0),
            result.get("llm_completion_tokens", 0),
        )
        chat_memory.save_user_memory(user_id, topic.strip(), result["draft"])
        await message.answer(
            _format_result_markdown(result),
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception:
        elapsed = time.perf_counter() - started_at
        agent_requests_failed_total.inc()
        agent_request_duration_seconds.observe(elapsed)
        logger.exception("Multi-agent graph execution failed for user_id=%s", user_id)
        await message.answer("Произошла ошибка, попробуйте позже.")
    finally:
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task


async def _typing_pulse(message: Message) -> None:
    """Send typing action periodically while long graph run is in progress."""
    while True:
        await message.answer_chat_action(action=ChatAction.TYPING)
        await asyncio.sleep(4)


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "Привет! Я Watchdog — твой AI-ассистент с мозгами 🧠\n"
        "Я не просто отвечаю на вопросы — я думаю, прежде чем сказать.\n"
        "За мной стоит команда агентов: Researcher ищет информацию,\n"
        "Writer формулирует ответ, а Reviewer проверяет качество.\n"
        "Напиши мне любой вопрос или используй /research <тема>\n"
        "для глубокого анализа.",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("research"))
async def research_command_handler(message: Message, command: CommandObject) -> None:
    topic = (command.args or "").strip()
    await _run_research_flow(message, topic)


@router.message(Command("clear"))
async def clear_history_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    deleted = chat_memory.clear_user_memory(user_id)
    await message.answer(
        f"История очищена. Удалено сообщений: {deleted}.",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("me"))
async def me_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    role = get_role(user_id)
    await message.answer(f"Твоя роль: `{role}` (user_id={user_id})", parse_mode=ParseMode.MARKDOWN)


@router.message(Command("setadmin"))
@require_role("owner")
async def set_admin_handler(message: Message, command: CommandObject) -> None:
    issuer_id = message.from_user.id if message.from_user else 0
    args = (command.args or "").strip()
    if not args.isdigit():
        await message.answer("Использование: `/setadmin <user_id>`", parse_mode=ParseMode.MARKDOWN)
        return
    target_id = int(args)
    set_role(target_id, "admin", granted_by=issuer_id)
    await message.answer(f"Пользователь `{target_id}` назначен админом.", parse_mode=ParseMode.MARKDOWN)


@router.message(Command("removeadmin"))
@require_role("owner")
async def remove_admin_handler(message: Message, command: CommandObject) -> None:
    args = (command.args or "").strip()
    if not args.isdigit():
        await message.answer(
            "Использование: `/removeadmin <user_id>`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    target_id = int(args)
    deleted = remove_role(target_id)
    if deleted:
        await message.answer(f"Права администратора сняты у `{target_id}`.", parse_mode=ParseMode.MARKDOWN)
    else:
        await message.answer(f"Для `{target_id}` не найдено admin-роли.", parse_mode=ParseMode.MARKDOWN)


@router.message(Command("admins"))
@require_role("owner")
async def list_admins_handler(message: Message) -> None:
    admins = list_admins()
    if not admins:
        await message.answer("Список администраторов пуст.")
        return
    lines = ["Список администраторов:"]
    for item in admins:
        lines.append(
            f"- user_id={item['user_id']} role={item['role']} by={item['granted_by']} at={item['granted_at']}"
        )
    await message.answer("\n".join(lines))


@router.message(Command("selftest"))
@require_role("admin")
async def selftest_handler(message: Message) -> None:
    if runtime_health_checker is not None:
        snapshot = await runtime_health_checker.run_once(allow_restart=False)
        await message.answer(
            "Selftest OK:\n"
            f"- db_ok: {snapshot.db_ok}\n"
            f"- telegram_ok: {snapshot.telegram_ok}\n"
            f"- memory_usage: {snapshot.memory_usage:.2f}%\n"
            f"- consecutive_telegram_failures: {snapshot.consecutive_telegram_failures}\n"
            f"- message: {snapshot.message}"
        )
        return

    checks = [
        f"DB users count: {count_users()}",
        f"DB conversations count: {count_conversations()}",
        f"LLM provider: {LLMConfig.get_provider()}",
        f"LLM model: {LLMConfig.get_model_name()}",
    ]
    await message.answer("Selftest OK:\n" + "\n".join(checks))


@router.message(Command("status"))
@require_role("admin")
async def status_handler(message: Message) -> None:
    uptime = int(time.time() - PROCESS_STARTED_AT)
    health = runtime_health_checker.last_snapshot if runtime_health_checker else None
    monitor = runtime_last_monitor_state
    await message.answer(
        "Status:\n"
        f"- pid: {os.getpid()}\n"
        f"- uptime_sec: {uptime}\n"
        f"- users: {count_users()}\n"
        f"- conversations: {count_conversations()}\n"
        f"- health_message: {health.message if health else 'n/a'}\n"
        f"- monitor_decision: {monitor['decision'] if monitor else 'n/a'}"
    )


@router.message(Command("fulldiag"))
@require_role("admin")
async def fulldiag_handler(message: Message) -> None:
    global runtime_last_monitor_state
    if runtime_health_checker is None:
        await message.answer("Self-diagnostics service is not initialized.")
        return
    snapshot = await runtime_health_checker.run_once(allow_restart=False)
    runtime_last_monitor_state = await run_monitor_agent(
        bot=message.bot,
        admin_user_id=runtime_health_checker.admin_user_id,
        health_snapshot=snapshot,
    )
    await message.answer(
        "Full diagnostics completed:\n"
        f"- decision: {runtime_last_monitor_state['decision']}\n"
        f"- analysis: {runtime_last_monitor_state['analysis']}\n"
        f"- alert_text: {runtime_last_monitor_state['alert_text']}"
    )


@router.message(Command("restart"))
@require_role("owner")
async def restart_handler(message: Message) -> None:
    await message.answer("Перезапускаюсь... 🚀")
    os.execv(sys.executable, [sys.executable, "-m", "telegram_bot.main"])


@router.message(lambda message: bool(message.text and not message.text.startswith("/")))
async def plain_text_handler(message: Message) -> None:
    await _run_research_flow(message, message.text or "")
