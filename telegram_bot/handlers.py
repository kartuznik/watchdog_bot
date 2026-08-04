"""Message handlers for Telegram + LangGraph integration."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import suppress
from urllib.parse import urlparse
from typing import cast

from aiogram import Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import InlineKeyboardMarkup, Message

try:
    from arq.connections import RedisSettings, create_pool
except Exception:  # pragma: no cover - optional in local dry-runs without deps installed.
    RedisSettings = None
    create_pool = None

from agents.database import (
    count_conversations,
    count_users,
    create_async_task,
    get_async_task,
    update_async_task_status,
)
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
from config import is_module_enabled
from telegram_bot.messaging import (
    OutgoingMessage,
    build_result_messages,
    chunk_text,
    format_user_facing_error,
    shorten_for_memory,
)
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


async def _answer_chunks(
    message: Message,
    parts: list[str] | list[OutgoingMessage],
    *,
    parse_mode: ParseMode | None = ParseMode.HTML,
) -> None:
    """Send one or more Telegram-safe chunks sequentially (HTML by default)."""
    for part in parts:
        markup: InlineKeyboardMarkup | None = None
        if isinstance(part, OutgoingMessage):
            text = part.text
            markup = part.reply_markup
        else:
            text = part
        if not str(text).strip():
            continue
        try:
            await message.answer(text, parse_mode=parse_mode, reply_markup=markup)
        except Exception:
            # HTML can fail on rare edge cases; retry as plain text without markup tags intent.
            await message.answer(text, parse_mode=None, reply_markup=markup)


def _redis_settings_from_env() -> RedisSettings | None:
    if RedisSettings is None:
        return None
    raw = os.getenv("REDIS_URL", "redis://redis:6379").strip()
    parsed = urlparse(raw)
    host = parsed.hostname or "redis"
    port = int(parsed.port or 6379)
    database = int((parsed.path or "/0").strip("/")) if (parsed.path or "").strip("/") else 0
    password = parsed.password
    return RedisSettings(host=host, port=port, database=database, password=password)


def _is_heavy_request(topic: str) -> bool:
    normalized = topic.strip().lower()
    if len(normalized) > 130:
        return True
    heavy_markers = (
        "исследуй",
        "сравни",
        "подробно",
        "нейросет",
        "проанализируй",
        "тенденц",
        "рынок",
    )
    return any(marker in normalized for marker in heavy_markers)


async def _enqueue_research_task(
    topic: str,
    user_id: int,
    conversation_history: list[dict[str, str]] | None = None,
) -> str | None:
    if create_pool is None:
        return None
    redis_settings = _redis_settings_from_env()
    if redis_settings is None:
        return None

    history = list(conversation_history or [])
    task_id = str(uuid.uuid4())
    create_async_task(
        task_id=task_id,
        user_id=user_id,
        task_type="research",
        payload=topic,
    )
    redis = await create_pool(redis_settings)
    try:
        await redis.enqueue_job(
            "process_research_task",
            topic,
            user_id,
            task_id,
            history,
            _job_id=task_id,
        )
    except Exception as exc:
        update_async_task_status(task_id, status="failed", error=str(exc))
        raise
    finally:
        await redis.close()
    return task_id


async def _poll_task_and_send_result(
    message: Message,
    task_id: str,
    *,
    topic: str,
    user_id: int,
) -> None:
    max_attempts = 180  # ~15 minutes with 5-second interval.
    for _ in range(max_attempts):
        task = get_async_task(task_id)
        if not task:
            await asyncio.sleep(5)
            continue
        status = str(task.get("status", "queued")).strip().lower()
        if status == "done":
            raw = str(task.get("result", "")).strip() or "Пустой ответ от worker."
            role = get_role(user_id)
            try:
                import json

                payload = json.loads(raw)
                if isinstance(payload, dict) and payload.get("draft") is not None:
                    outgoing = build_result_messages(
                        payload,
                        viewer_user_id=user_id,
                        viewer_role=role,
                    )
                    chat_memory.save_user_memory(
                        user_id,
                        topic,
                        shorten_for_memory([m.text for m in outgoing]),
                    )
                    await _answer_chunks(message, outgoing)
                    return
            except json.JSONDecodeError:
                pass
            chat_memory.save_user_memory(user_id, topic, raw)
            parts = chunk_text(f"✅ Фоновая задача завершена:\n\n{raw}")
            await _answer_chunks(message, parts, parse_mode=None)
            return
        if status == "failed":
            error = str(task.get("error", "")).strip() or "неизвестная ошибка"
            # Keep worker errors honest but avoid leaking raw secrets/keys.
            safe = format_user_facing_error(RuntimeError(error))
            await message.answer(f"❌ Фоновая задача завершилась ошибкой.\n{safe}")
            return
        await asyncio.sleep(5)
    await message.answer(
        "⏱ Фоновая задача всё ещё выполняется. Проверь позже командой `/status` или повтори запрос."
    )


async def _run_research_flow(message: Message, topic: str) -> None:
    if not topic.strip():
        await message.answer(
            "Укажи тему после команды.\n"
            "Пример: <code>/research агенты в поддержке клиентов</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    user_id = message.from_user.id if message.from_user else 0
    conversation_history = chat_memory.get_user_memory(user_id)

    if is_module_enabled("background_worker") and _is_heavy_request(topic):
        try:
            task_id = await _enqueue_research_task(
                topic.strip(),
                user_id,
                conversation_history,
            )
        except Exception:
            task_id = None
            logger.exception("Failed to enqueue heavy research task")
        if task_id:
            await message.answer(
                f"Задача принята, обрабатываю ⏳\nID: <code>{task_id}</code>",
                parse_mode=ParseMode.HTML,
            )
            asyncio.create_task(
                _poll_task_and_send_result(
                    message,
                    task_id,
                    topic=topic.strip(),
                    user_id=user_id,
                )
            )
            return

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
            cost_usd=float(result.get("estimated_cost_usd", 0.0) or 0.0),
        )
        role = get_role(user_id)
        outgoing = build_result_messages(
            dict(result),
            viewer_user_id=user_id,
            viewer_role=role,
        )
        chat_memory.save_user_memory(
            user_id,
            topic.strip(),
            shorten_for_memory([m.text for m in outgoing]),
        )
        await _answer_chunks(message, outgoing)
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        agent_requests_failed_total.inc()
        agent_request_duration_seconds.observe(elapsed)
        logger.exception("Multi-agent graph execution failed for user_id=%s", user_id)
        await message.answer(format_user_facing_error(exc))
    finally:
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task


async def _typing_pulse(message: Message) -> None:
    """Send typing action periodically while long graph run is in progress."""
    chat_id = message.chat.id
    bot = message.bot
    while True:
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
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
