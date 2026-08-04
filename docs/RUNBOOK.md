# Watchdog Bot Runbook

Operational guide for self-hosted deployments. No secrets, hostnames, or IPs are documented here — use your environment’s `.env` and compose project.

## Deploy

1. Clone the repository and create `.env` from `.env.example`.
2. Required: `TELEGRAM_BOT_TOKEN`, at least one LLM key (`OPENAI_API_KEY` and/or `DEEPSEEK_API_KEY`), `ADMIN_PASSWORD`.
3. Recommended: `OWNER_ID`, `TAVILY_API_KEY`, `TELEGRAM_ALERT_CHAT_ID` (usually same as `OWNER_ID`), `GRAFANA_ADMIN_PASSWORD`.
4. Start stack:

```bash
docker compose up -d --build
docker compose ps
```

5. Verify:
   - Bot polling logs are healthy (`docker compose logs --tail=100 bot`).
   - Metrics scrape target is up in Prometheus.
   - Admin panel answers Basic Auth on the admin service port.
   - Grafana loads the **Watchdog Bot Overview** dashboard (folder **Watchdog**) and alerting contact point `telegram-owner`.

After code changes that affect runtime, rebuild only the touched services, for example:

```bash
docker compose up -d --build bot worker admin-panel grafana
```

## Backup (database and configuration)

### SQLite memory

- Default path inside containers: `/app/data/agent_memory.db` (host bind: `./data`).
- WAL mode is enabled; for a consistent copy stop writers briefly or use SQLite backup API.

Suggested cold backup:

```bash
docker compose stop bot worker admin-panel
cp ./data/agent_memory.db ./backups/agent_memory-$(date +%Y%m%d).db
# also copy -wal/-shm if present while stopped
docker compose start bot worker admin-panel
```

### Configuration

- Back up `.env` out-of-band (secrets manager / encrypted store). Never commit it.
- Compose and `monitoring/` are in git; restore by redeploying the same revision.

### Retention

- `DATA_RETENTION_DAYS` (default `90`) controls hard purge of soft-deleted rows and aged `usage_events`.
- Admin: `POST /api/retention/purge` (Basic Auth).

## API key rotation

Rotate one provider at a time; keep the other key valid so LLM fallback can cover the cutover.

| Secret | Steps |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Issue new token in BotFather → update `.env` → recreate `bot` (and Grafana if alerts use the same token). Old token invalidates polling. |
| `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | Create new key → update `.env` → recreate `bot` and `worker` → revoke old key after smoke. |
| `TAVILY_API_KEY` | Update `.env` → recreate `bot` and `worker`. Invalid keys soft-fail search only. |
| `ADMIN_PASSWORD` | Update `.env` → recreate `admin-panel`. |
| `GRAFANA_ADMIN_PASSWORD` | Update `.env` → recreate `grafana`. |
| `TELEGRAM_ALERT_CHAT_ID` | Set to the owner chat/user id → recreate `grafana`. |

Never paste live keys into git, issues, or chat logs.

## Incident response

### High error rate (Grafana: HighErrorRate)

- Threshold: failed / (failed + success) **> 5%** on a **5m** window.
- Check bot/worker logs, LLM provider status, and recent deploys.
- Confirm metrics: `agent_requests_failed_total`, `agent_requests_total`.
- Mitigate: roll back image/revision, fix provider keys, temporarily disable heavy modules via `ENABLED_MODULES`.

### Async queue lag (Grafana: AsyncQueueLag)

- Threshold: `agent_async_queue_lag_seconds` **> 300** (5 minutes).
- Check Redis health, `worker` logs, and `async_tasks` statuses (`queued` / `running`).
- Mitigate: scale/restart `worker`, clear stuck jobs carefully, reduce heavy-request load.

### LLM provider fallback (Grafana: LLMProviderFallback)

- Threshold: `sum(increase(agent_llm_fallback_total[10m]))` **> 0**.
- Primary provider auth/balance/network failed; secondary was used.
- Mitigate: restore primary key/balance; confirm both providers in `.env`; watch `agent_llm_fallback_total`.

### Bot down / Telegram conflicts

- Single polling instance per bot token.
- Health/monitor loops (when `self_diagnostics` enabled) can notify `OWNER_ID` independently of Grafana.

### Soft-delete / data requests

- Default `/clear` and admin clear are **soft** (`deleted_at`).
- Admin soft-delete user: `POST /api/users/{id}/soft_delete`.
- Optional gate `SOFT_DELETE_GATE=true` blocks soft-deleted users from the bot (off by default).
- Hard purge only via retention job or `hard=true` on clear (use sparingly).

## Rollback

1. Identify last known-good git revision.
2. `git checkout <revision>` (or redeploy the previous image tag).
3. `docker compose up -d --build` for affected services.
4. Restore SQLite from backup if the schema/data migration is incompatible.
5. Smoke: creative (no search) + factual (search) Telegram requests; hit admin `/api/stats` and `/api/usage`.

## Alerting notes

- Grafana contact point uses `$__env{TELEGRAM_BOT_TOKEN}` and `$__env{TELEGRAM_ALERT_CHAT_ID}` — no tokens in YAML.
- Unified alerting must stay enabled (`GF_UNIFIED_ALERTING_ENABLED=true`).
- In-bot health alerts to `OWNER_ID` remain a complementary path when Grafana is unavailable.
