"""Admin panel usage/export/soft-delete API tests."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents import database as database_module
from admin_panel import app as admin_app_module


@pytest.fixture()
def admin_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_path = tmp_path / "admin_memory.db"
    monkeypatch.setenv("AGENT_DB_PATH", str(db_path))
    monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-pass")
    monkeypatch.setenv("DATA_RETENTION_DAYS", "90")
    monkeypatch.setattr(database_module, "DB_PATH", db_path)
    database_module.init_db()
    database_module.record_usage_event(
        user_id=42,
        prompt_tokens=10,
        completion_tokens=5,
        estimated_cost_usd=0.0003,
        router_decision="no_search",
    )
    with database_module.get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            ("42", "user", "hello admin"),
        )
        conn.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            ("42", "assistant", "hi"),
        )
        conn.commit()
    return TestClient(admin_app_module.app)


def _auth_header(password: str = "test-admin-pass") -> dict[str, str]:
    token = base64.b64encode(f"admin:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_usage_endpoint(admin_client: TestClient) -> None:
    resp = admin_client.get("/api/usage", headers=_auth_header())
    assert resp.status_code == 200
    rows = resp.json()
    assert any(str(r["user_id"]) == "42" for r in rows)
    row = next(r for r in rows if str(r["user_id"]) == "42")
    assert row["requests_count"] == 1
    assert row["prompt_tokens"] == 10


def test_export_json_and_csv(admin_client: TestClient) -> None:
    js = admin_client.get(
        "/api/export/dialogs",
        params={"user_id": 42, "format": "json"},
        headers=_auth_header(),
    )
    assert js.status_code == 200
    assert "application/json" in js.headers["content-type"]
    assert "hello admin" in js.text

    csv_resp = admin_client.get(
        "/api/export/dialogs",
        params={"user_id": 42, "format": "csv"},
        headers=_auth_header(),
    )
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "hello admin" in csv_resp.text
    assert "role" in csv_resp.text.splitlines()[0]


def test_soft_delete_and_clear_memory_soft(admin_client: TestClient) -> None:
    sd = admin_client.post("/api/users/42/soft_delete", headers=_auth_header())
    assert sd.status_code == 200
    body = sd.json()
    assert body["conversations_marked"] >= 1
    assert database_module.is_user_soft_deleted(42)

    # Soft clear on empty active set is fine.
    cleared = admin_client.post("/api/clear_memory", headers=_auth_header())
    assert cleared.status_code == 200
    assert cleared.json()["hard"] is False


def test_unauthorized(admin_client: TestClient) -> None:
    resp = admin_client.get("/api/usage")
    assert resp.status_code == 401
