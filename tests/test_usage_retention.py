"""usage_events aggregation, soft-delete and retention purge."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import database as database_module


@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "usage_memory.db"
    monkeypatch.setenv("AGENT_DB_PATH", str(db_path))
    monkeypatch.setenv("DATA_RETENTION_DAYS", "90")
    monkeypatch.setattr(database_module, "DB_PATH", db_path)
    database_module.init_db()
    return db_path


def test_record_and_aggregate_usage(temp_db: Path) -> None:
    database_module.record_usage_event(
        user_id=101,
        prompt_tokens=100,
        completion_tokens=50,
        estimated_cost_usd=0.001,
        router_decision="no_search",
    )
    database_module.record_usage_event(
        user_id=101,
        prompt_tokens=20,
        completion_tokens=10,
        estimated_cost_usd=0.0002,
        router_decision="search",
    )
    database_module.record_usage_event(
        user_id=202,
        prompt_tokens=5,
        completion_tokens=5,
        estimated_cost_usd=0.0001,
        router_decision="search",
    )
    rows = database_module.aggregate_usage_by_user()
    by_user = {str(r["user_id"]): r for r in rows}
    assert by_user["101"]["requests_count"] == 2
    assert by_user["101"]["prompt_tokens"] == 120
    assert by_user["101"]["completion_tokens"] == 60
    assert abs(float(by_user["101"]["estimated_cost_usd"]) - 0.0012) < 1e-9
    assert by_user["202"]["requests_count"] == 1


def test_soft_delete_hides_conversations(temp_db: Path) -> None:
    database_module.ensure_user(7)
    with database_module.get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            ("7", "user", "hello"),
        )
        conn.commit()
    assert len(database_module.list_recent_conversations(user_id=7)) == 1
    marked = database_module.soft_delete_conversations(user_id=7)
    assert marked == 1
    assert database_module.list_recent_conversations(user_id=7) == []
    assert (
        len(
            database_module.list_recent_conversations(
                user_id=7, include_deleted=True
            )
        )
        == 1
    )


def test_soft_delete_user_and_retention_purge(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_module.ensure_user(9)
    with database_module.get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            ("9", "user", "secret"),
        )
        conn.commit()
    result = database_module.soft_delete_user(9)
    assert result["conversations_marked"] == 1
    assert database_module.is_user_soft_deleted(9)

    # Age soft-deleted rows beyond retention for purge.
    with database_module.get_connection() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET deleted_at=datetime('now', '-120 days')
            WHERE user_id='9'
            """
        )
        conn.execute(
            """
            UPDATE users
            SET deleted_at=datetime('now', '-120 days')
            WHERE user_id='9'
            """
        )
        conn.execute(
            """
            INSERT INTO usage_events (
                user_id, prompt_tokens, completion_tokens, estimated_cost_usd, router_decision, created_at
            )
            VALUES ('9', 1, 1, 0.0, 'search', datetime('now', '-120 days'))
            """
        )
        conn.commit()

    monkeypatch.setenv("DATA_RETENTION_DAYS", "90")
    purged = database_module.purge_expired_data(retention_days=90)
    assert purged["conversations_purged"] >= 1
    assert purged["users_purged"] >= 1
    assert purged["usage_events_purged"] >= 1
    assert database_module.list_recent_conversations(
        user_id=9, include_deleted=True
    ) == []


def test_clear_conversations_soft_by_default(temp_db: Path) -> None:
    with database_module.get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            ("1", "user", "x"),
        )
        conn.commit()
    n = database_module.clear_conversations(user_id=1)
    assert n == 1
    assert database_module.list_recent_conversations(user_id=1) == []
    remaining = database_module.list_recent_conversations(
        user_id=1, include_deleted=True
    )
    assert len(remaining) == 1
