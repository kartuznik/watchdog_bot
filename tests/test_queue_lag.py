"""Async queue lag helper tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import database as database_module
from agents.metrics import (
    agent_async_queue_lag_seconds,
    refresh_async_queue_lag,
    set_async_queue_lag_seconds,
)


@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "lag_memory.db"
    monkeypatch.setenv("AGENT_DB_PATH", str(db_path))
    monkeypatch.setattr(database_module, "DB_PATH", db_path)
    database_module.init_db()
    return db_path


def test_compute_lag_zero_without_pending_tasks(temp_db: Path) -> None:
    assert database_module.compute_async_queue_lag_seconds() == 0.0


def test_compute_lag_for_queued_task(temp_db: Path) -> None:
    database_module.create_async_task(
        task_id="t1",
        user_id=1,
        task_type="research",
        payload="topic",
        stage="queued",
    )
    with database_module.get_connection() as conn:
        conn.execute(
            """
            UPDATE async_tasks
            SET created_at=datetime('now', '-120 seconds')
            WHERE task_id='t1'
            """
        )
        conn.commit()
    lag = database_module.compute_async_queue_lag_seconds()
    assert lag >= 100.0


def test_refresh_async_queue_lag_sets_gauge(temp_db: Path) -> None:
    set_async_queue_lag_seconds(0.0)
    database_module.create_async_task(
        task_id="t2",
        user_id=2,
        task_type="research",
        payload="heavy",
    )
    with database_module.get_connection() as conn:
        conn.execute(
            """
            UPDATE async_tasks
            SET status='running', created_at=datetime('now', '-45 seconds')
            WHERE task_id='t2'
            """
        )
        conn.commit()
    lag = refresh_async_queue_lag()
    assert lag >= 30.0
    # Gauge sample should reflect a positive lag.
    samples = list(agent_async_queue_lag_seconds.collect())[0].samples
    values = [s.value for s in samples if s.name == "agent_async_queue_lag_seconds"]
    assert values and values[0] >= 30.0
