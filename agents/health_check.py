"""Lightweight health checker for watchdog-style self-healing."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot

from agents.database import DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    db_ok: bool
    telegram_ok: bool
    memory_usage: float
    consecutive_telegram_failures: int
    restarted: bool
    message: str


class HealthCheckService:
    """Non-LLM fallback checker running every N seconds."""

    def __init__(
        self,
        *,
        bot: Bot,
        admin_user_id: int | None,
        db_path: Path = DB_PATH,
    ) -> None:
        self.bot = bot
        self.admin_user_id = admin_user_id
        self.db_path = db_path
        self.consecutive_telegram_failures = 0
        self.last_snapshot = HealthSnapshot(
            db_ok=True,
            telegram_ok=True,
            memory_usage=0.0,
            consecutive_telegram_failures=0,
            restarted=False,
            message="Health checker initialized.",
        )

    async def run_once(self, *, allow_restart: bool = False) -> HealthSnapshot:
        db_ok = self._check_db_ping()
        telegram_ok = await self._check_telegram()
        memory_usage = self._read_memory_usage_percent()
        restarted = False

        problems: list[str] = []
        if not db_ok:
            problems.append("DB ping failed")
        if not telegram_ok:
            problems.append("Telegram API check failed")
        if memory_usage > 90:
            problems.append(f"High memory usage: {memory_usage:.2f}%")

        if telegram_ok:
            self.consecutive_telegram_failures = 0
        else:
            self.consecutive_telegram_failures += 1

        if self.consecutive_telegram_failures >= 3:
            problems.append("3 consecutive Telegram failures")
            if allow_restart:
                await self._send_alert("Critical: restarting process after 3 Telegram failures.")
                restarted = True
                self._restart_process()

        if memory_usage > 90:
            await self._send_alert(f"Warning: memory usage is high ({memory_usage:.2f}%).")
        if not db_ok:
            await self._send_alert("Critical: DB ping failed. Check SQLite availability.")

        message = "OK" if not problems else "; ".join(problems)
        snapshot = HealthSnapshot(
            db_ok=db_ok,
            telegram_ok=telegram_ok,
            memory_usage=memory_usage,
            consecutive_telegram_failures=self.consecutive_telegram_failures,
            restarted=restarted,
            message=message,
        )
        self.last_snapshot = snapshot
        logger.info("Health check snapshot: %s", snapshot)
        return snapshot

    async def run_forever(self, interval_seconds: int = 60) -> None:
        while True:
            try:
                await self.run_once(allow_restart=True)
            except Exception:
                logger.exception("Health checker loop failed unexpectedly")
            await asyncio.sleep(max(5, interval_seconds))

    def _check_db_ping(self) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("SELECT 1").fetchone()
            finally:
                conn.close()
            return True
        except Exception:
            logger.exception("DB ping failed for path=%s", self.db_path)
            return False

    async def _check_telegram(self) -> bool:
        try:
            await self.bot.get_me()
            return True
        except Exception:
            logger.exception("Telegram get_me failed")
            return False

    def _read_memory_usage_percent(self) -> float:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                data = f.read()
            total_kb = self._extract_kb(data, "MemTotal")
            available_kb = self._extract_kb(data, "MemAvailable")
            if total_kb <= 0:
                return 0.0
            used_kb = max(0, total_kb - available_kb)
            return (used_kb / total_kb) * 100.0
        except Exception:
            logger.exception("Failed to read memory usage")
            return 0.0

    @staticmethod
    def _extract_kb(meminfo_text: str, key: str) -> int:
        prefix = f"{key}:"
        for line in meminfo_text.splitlines():
            if line.startswith(prefix):
                raw = line.split(":", 1)[1].strip().split()[0]
                try:
                    return int(raw)
                except ValueError:
                    return 0
        return 0

    async def _send_alert(self, text: str) -> None:
        if not self.admin_user_id:
            logger.warning("Health alert skipped (OWNER_ID is not set): %s", text)
            return
        try:
            await self.bot.send_message(chat_id=self.admin_user_id, text=f"[Health Alert] {text}")
        except Exception:
            logger.exception("Failed to send health alert to owner")

    def _restart_process(self) -> None:
        logger.error("Restarting process via os.execv after critical health state")
        os.execv(sys.executable, [sys.executable, "-m", "telegram_bot.main"])
