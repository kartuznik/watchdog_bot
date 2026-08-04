"""SQLite connection hardening tests (WAL + busy_timeout)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import database as database_module


@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "test_agent_memory.db"
    monkeypatch.setenv("AGENT_DB_PATH", str(db_path))
    monkeypatch.setattr(database_module, "DB_PATH", db_path)
    database_module.init_db()
    return db_path


def test_get_connection_enables_wal_and_busy_timeout(temp_db: Path) -> None:
    with database_module.get_connection() as conn:
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert str(journal).lower() == "wal"
    assert int(busy) >= 30000
    assert temp_db.exists()
