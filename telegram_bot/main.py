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
from agents.roles import get_owner_id, set_role
from telegram_bot.handlers import router
from telegram_bot.middlewares.role_check import RoleCheckMiddleware


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

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
