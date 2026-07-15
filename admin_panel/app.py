"""FastAPI admin panel adapted from rag-telegram-bot/admin_panel/app.py."""

from __future__ import annotations

import os
import secrets
import time

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from agents.anchors import list_user_anchors
from agents.database import clear_conversations, init_db, list_anchors, list_recent_conversations

app = FastAPI(title="AI Agents Lab Admin Panel")
security = HTTPBasic()
started_at = time.time()


def _admin_password() -> str:
    return os.getenv("ADMIN_PASSWORD", "change-me")


def _authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    is_valid_user = secrets.compare_digest(credentials.username, "admin")
    is_valid_password = secrets.compare_digest(credentials.password, _admin_password())
    if not (is_valid_user and is_valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _format_uptime(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


@app.on_event("startup")
async def _startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(_: str = Depends(_authenticate)) -> str:
    return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Agents Lab Admin</title>
  <style>
    body { font-family: Arial, sans-serif; background:#111827; color:#e5e7eb; margin:0; padding:24px; }
    h1 { margin:0 0 16px 0; }
    .grid { display:grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:20px; }
    .card { background:#1f2937; border:1px solid #374151; border-radius:12px; padding:14px; }
    .label { color:#9ca3af; font-size:13px; }
    .value { font-size:24px; font-weight:700; margin-top:6px; }
    button { background:#2563eb; color:white; border:none; border-radius:8px; padding:10px 14px; cursor:pointer; }
    button:hover { background:#1d4ed8; }
    table { width:100%; border-collapse: collapse; margin-top:10px; }
    th, td { border-bottom:1px solid #374151; text-align:left; padding:8px; font-size:14px; }
    .muted { color:#9ca3af; font-size:13px; }
  </style>
</head>
<body>
  <h1>AI Agents Lab Admin</h1>
  <p class="muted">Статистика, память и якоря.</p>
  <div class="grid">
    <div class="card"><div class="label">Диалоги</div><div id="stat-conv" class="value">-</div></div>
    <div class="card"><div class="label">Якоря</div><div id="stat-anchors" class="value">-</div></div>
    <div class="card"><div class="label">Аптайм</div><div id="stat-uptime" class="value">-</div></div>
  </div>
  <div class="card">
    <button onclick="clearMemory()">Очистить память</button>
    <div id="status" class="muted" style="margin-top:8px;"></div>
  </div>
  <div class="card" style="margin-top:12px;">
    <h3>Последние диалоги</h3>
    <table><thead><tr><th>ID</th><th>User</th><th>Role</th><th>Content</th><th>Created</th></tr></thead><tbody id="conversations"></tbody></table>
  </div>
  <script>
    async function refresh() {
      const conversations = await fetch('/api/conversations').then(r => r.json());
      const anchors = await fetch('/api/anchors').then(r => r.json());
      const uptime = await fetch('/api/uptime').then(r => r.json());

      document.getElementById('stat-conv').textContent = conversations.length;
      document.getElementById('stat-anchors').textContent = anchors.length;
      document.getElementById('stat-uptime').textContent = uptime.uptime;

      const tbody = document.getElementById('conversations');
      tbody.innerHTML = '';
      for (const row of conversations) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.id}</td><td>${row.user_id}</td><td>${row.role}</td><td>${String(row.content).slice(0,120)}</td><td>${row.created_at}</td>`;
        tbody.appendChild(tr);
      }
    }

    async function clearMemory() {
      const result = await fetch('/api/clear_memory', { method: 'POST' }).then(r => r.json());
      document.getElementById('status').textContent = result.message;
      await refresh();
    }

    refresh();
    setInterval(refresh, 10000);
  </script>
</body>
</html>"""


@app.get("/api/uptime")
async def api_uptime(_: str = Depends(_authenticate)) -> dict[str, str]:
    return {"uptime": _format_uptime(int(time.time() - started_at))}


@app.get("/api/conversations")
async def api_conversations(
    _: str = Depends(_authenticate),
    user_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    return list_recent_conversations(limit=limit, user_id=user_id)


@app.post("/api/clear_memory")
async def api_clear_memory(
    _: str = Depends(_authenticate),
    user_id: int | None = Query(default=None),
) -> dict[str, str | int]:
    deleted = clear_conversations(user_id=user_id)
    return {"message": f"Memory cleared. Deleted rows: {deleted}", "deleted": deleted}


@app.get("/api/anchors")
async def api_anchors(
    _: str = Depends(_authenticate),
    user_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    if user_id is not None:
        return list_user_anchors(user_id=user_id, limit=limit)
    return list_anchors(limit=limit, user_id=None)


if __name__ == "__main__":
    uvicorn.run("admin_panel.app:app", host="0.0.0.0", port=8004)
