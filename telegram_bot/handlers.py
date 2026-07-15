"""Message handlers for Telegram + LangGraph integration."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import cast

from aiogram import Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

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

logger = logging.getLogger(__name__)
router = Router()
multi_agent_graph = build_multi_agent_graph()
chat_memory = ChatMemory(max_messages=20)


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


@router.message(lambda message: bool(message.text and not message.text.startswith("/")))
async def plain_text_handler(message: Message) -> None:
    await _run_research_flow(message, message.text or "")
