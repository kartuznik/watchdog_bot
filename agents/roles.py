"""Role-based access utilities inspired by SurveyBot role model."""

from __future__ import annotations

import os
from typing import Literal

from agents.database import get_connection

RoleName = Literal["owner", "admin", "user"]
ROLE_PRIORITY: dict[str, int] = {"user": 1, "admin": 2, "owner": 3}


def get_owner_id() -> int | None:
    raw = os.getenv("OWNER_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def normalize_role(value: str) -> RoleName:
    role = value.strip().lower()
    if role not in {"owner", "admin", "user"}:
        raise ValueError("Role must be one of: owner, admin, user")
    return role  # type: ignore[return-value]


def get_role(user_id: int) -> RoleName:
    owner_id = get_owner_id()
    if owner_id is not None and int(user_id) == owner_id:
        return "owner"

    with get_connection() as conn:
        row = conn.execute(
            "SELECT role FROM roles WHERE user_id=?",
            (str(user_id),),
        ).fetchone()
    if not row or not row["role"]:
        return "user"
    role = str(row["role"]).strip().lower()
    return role if role in ROLE_PRIORITY else "user"  # type: ignore[return-value]


def has_required_role(user_id: int, required_role: RoleName) -> bool:
    current_role = get_role(user_id)
    return ROLE_PRIORITY[current_role] >= ROLE_PRIORITY[required_role]


def set_role(user_id: int, role: RoleName, granted_by: int) -> None:
    normalized = normalize_role(role)
    owner_id = get_owner_id()
    if owner_id is not None and int(user_id) == owner_id and normalized != "owner":
        raise ValueError("Owner role cannot be downgraded")
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO roles (user_id, role, granted_by, granted_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                role=excluded.role,
                granted_by=excluded.granted_by,
                granted_at=CURRENT_TIMESTAMP
            """,
            (str(user_id), normalized, str(granted_by)),
        )
        conn.commit()


def remove_role(user_id: int) -> int:
    owner_id = get_owner_id()
    if owner_id is not None and int(user_id) == owner_id:
        raise ValueError("Owner role cannot be removed")
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM roles WHERE user_id=?", (str(user_id),))
        conn.commit()
    return int(cur.rowcount or 0)


def list_admins() -> list[dict[str, str]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, role, granted_by, granted_at
            FROM roles
            WHERE role IN ('admin', 'owner')
            ORDER BY role DESC, granted_at DESC
            """
        ).fetchall()
    result = [dict(r) for r in rows]

    owner_id = get_owner_id()
    if owner_id is not None and not any(str(owner_id) == row["user_id"] for row in result):
        result.insert(
            0,
            {
                "user_id": str(owner_id),
                "role": "owner",
                "granted_by": str(owner_id),
                "granted_at": "env",
            },
        )
    return result


def list_roles(limit: int = 1000) -> list[dict[str, str]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, role, granted_by, granted_at
            FROM roles
            ORDER BY granted_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 5000)),),
        ).fetchall()
    result = [dict(r) for r in rows]
    owner_id = get_owner_id()
    if owner_id is not None and not any(str(owner_id) == row["user_id"] for row in result):
        result.insert(
            0,
            {
                "user_id": str(owner_id),
                "role": "owner",
                "granted_by": str(owner_id),
                "granted_at": "env",
            },
        )
    return result
