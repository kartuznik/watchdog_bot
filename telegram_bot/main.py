"""Entry point for aiogram 3 bot with LangGraph integration."""

from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv
from prometheus_client import start_http_server

from agents.database import init_db
from agents.health_check import HealthCheckService
from agents.monitor_agent import run_monitor_agent
from agents.roles import get_owner_id, set_role
from config import is_module_enabled
from telegram_bot.handlers import configure_runtime_services, router, set_last_monitor_state
from telegram_bot.middlewares.role_check import RoleCheckMiddleware


logger = logging.getLogger(__name__)


async def _monitor_loop(
    *,
    bot: Bot,
    health_checker: HealthCheckService,
    interval_seconds: int = 600,
) -> None:
    while True:
        try:
            state = await run_monitor_agent(
                bot=bot,
                admin_user_id=health_checker.admin_user_id,
                health_snapshot=health_checker.last_snapshot,
            )
            set_last_monitor_state(state)
            logger.info("Monitor loop decision: %s", state["decision"])
        except Exception:
            logger.exception("Monitor loop iteration failed")
        await asyncio.sleep(max(60, interval_seconds))


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    init_db()
    owner_id = get_owner_id()
    if owner_id is not None:
        try:
            set_role(owner_id, "owner", granted_by=owner_id)
        except Exception:
            logging.getLogger(__name__).exception("Failed to enforce owner role")
    metrics_port = int(os.getenv("METRICS_PORT", "8001"))
    start_http_server(metrics_port)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Add it to ai-agents-lab/.env."
        )

    bot = Bot(
        token=bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )
    dp = Dispatcher()
    dp.message.middleware(RoleCheckMiddleware())
    dp.include_router(router)

    background_tasks: list[asyncio.Task] = []
    if is_module_enabled("self_diagnostics"):
        health_checker = HealthCheckService(bot=bot, admin_user_id=owner_id)
        configure_runtime_services(health_checker=health_checker)
        background_tasks.append(asyncio.create_task(health_checker.run_forever(interval_seconds=60)))
        background_tasks.append(
            asyncio.create_task(
                _monitor_loop(bot=bot, health_checker=health_checker, interval_seconds=600)
            )
        )
    else:
        configure_runtime_services(health_checker=None)

    try:
        await dp.start_polling(bot)
    finally:
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
