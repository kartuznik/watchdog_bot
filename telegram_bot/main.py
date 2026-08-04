"""Entry point for aiogram 3 bot with LangGraph integration."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv
from prometheus_client import start_http_server

from agents.database import init_db
from agents.health_check import HealthCheckService
from agents.metrics import ensure_baseline_metric_series, refresh_async_queue_lag
from agents.monitor_agent import run_monitor_agent
from agents.roles import get_owner_id, set_role
from config import is_module_enabled
from telegram_bot.handlers import configure_runtime_services, router, set_last_monitor_state
from telegram_bot.middlewares.role_check import RoleCheckMiddleware


logger = logging.getLogger(__name__)


def _smoke_request_path() -> Path:
    raw = os.getenv("AGENT_DB_PATH", "").strip()
    base = Path(raw).expanduser().resolve().parent if raw else Path("data").resolve()
    return base / "ops_smoke.json"


async def _queue_lag_loop(
    *,
    bot: Bot,
    owner_id: int | None,
    interval_seconds: int = 30,
) -> None:
    """Publish async queue lag gauge; also drain optional ops_smoke.json in-process."""
    while True:
        try:
            lag = refresh_async_queue_lag()
            logger.debug("async_queue_lag_seconds=%.1f", lag)
        except Exception:
            logger.exception("Failed to refresh async queue lag metric")
        try:
            await _drain_ops_smoke(bot=bot, owner_id=owner_id)
        except Exception:
            logger.exception("ops smoke drain failed")
        await asyncio.sleep(max(5, interval_seconds))


async def _drain_ops_smoke(*, bot: Bot, owner_id: int | None) -> None:
    """Run one in-process research topic from data/ops_smoke.json (main metrics registry)."""
    if owner_id is None:
        return
    path = _smoke_request_path()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)
    topic = str((payload or {}).get("topic", "")).strip()
    if not topic:
        return
    from telegram_bot.handlers import _run_research_flow

    class _User:
        def __init__(self, uid: int) -> None:
            self.id = uid

    class _Chat:
        def __init__(self, cid: int) -> None:
            self.id = cid

    class _Message:
        def __init__(self) -> None:
            self.bot = bot
            self.chat = _Chat(owner_id)
            self.from_user = _User(owner_id)
            self.text = topic

        async def answer(self, text, **kwargs):
            allowed = {
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "parse_mode",
                    "reply_markup",
                    "link_preview_options",
                    "disable_web_page_preview",
                }
            }
            return await bot.send_message(self.chat.id, text, **allowed)

    logger.info("ops smoke: running topic in main process")
    await _run_research_flow(_Message(), topic)


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
    ensure_baseline_metric_series()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Add it to .env."
        )

    bot = Bot(
        token=bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )
    dp = Dispatcher()
    dp.message.middleware(RoleCheckMiddleware())
    dp.include_router(router)

    background_tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _queue_lag_loop(bot=bot, owner_id=owner_id, interval_seconds=15)
        ),
    ]
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
