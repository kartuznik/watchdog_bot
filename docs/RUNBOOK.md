# Watchdog Bot — Runbook

Операционное руководство для self-hosted деплоя. Секреты, hostname и IP сюда не пишем — используйте `.env` и Compose-проект окружения.

## Deploy

1. Клонируйте репозиторий и создайте `.env` из `.env.example`.
2. Обязательно: `TELEGRAM_BOT_TOKEN`, хотя бы один LLM-ключ (`OPENAI_API_KEY` и/или `DEEPSEEK_API_KEY`), `ADMIN_PASSWORD`.
3. Рекомендуется: `OWNER_ID`, `TAVILY_API_KEY`, `TELEGRAM_ALERT_CHAT_ID` (обычно тот же, что `OWNER_ID`), `GRAFANA_ADMIN_PASSWORD`.
4. Запуск стека:

```bash
docker compose up -d --build
docker compose ps
```

5. Проверка:
   - Логи polling бота здоровы (`docker compose logs --tail=100 bot`).
   - Цель scrape в Prometheus в состоянии up.
   - Admin panel отвечает Basic Auth на порту admin-сервиса.
   - Grafana открывает дашборд **Watchdog Bot Overview** (папка **Watchdog**) и contact point `telegram-owner`.

После изменений кода, влияющих на runtime, пересобирайте только затронутые сервисы, например:

```bash
docker compose up -d --build bot worker admin-panel grafana
```

## Backup (база и конфигурация)

### SQLite (память агента)

- Путь по умолчанию внутри контейнеров: `/app/data/agent_memory.db` (host bind: `./data`).
- Включён WAL; для согласованной копии кратко остановите writers или используйте SQLite backup API.

Холодный backup:

```bash
docker compose stop bot worker admin-panel
cp ./data/agent_memory.db ./backups/agent_memory-$(date +%Y%m%d).db
# также скопируйте -wal/-shm, если они есть, пока сервисы остановлены
docker compose start bot worker admin-panel
```

### Конфигурация

- `.env` храните вне git (secrets manager / шифрованное хранилище). Никогда не коммитьте.
- Compose и `monitoring/` в git; восстановление — повторный деплой той же ревизии.

### Retention

- `DATA_RETENTION_DAYS` (по умолчанию `90`) управляет hard purge soft-deleted строк и устаревших `usage_events`.
- Admin: `POST /api/retention/purge` (Basic Auth).

## Ротация API-ключей

Ротируйте одного провайдера за раз; второй ключ держите валидным, чтобы LLM fallback перекрыл cutover.

| Секрет | Шаги |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Новый токен в BotFather → обновить `.env` → recreate `bot` (и Grafana, если алерты используют тот же токен). Старый токен инвалидирует polling. |
| `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | Создать новый ключ → обновить `.env` → recreate `bot` и `worker` → отозвать старый ключ после smoke. |
| `TAVILY_API_KEY` | Обновить `.env` → recreate `bot` и `worker`. Невалидный ключ — soft-fail только поиска. |
| `ADMIN_PASSWORD` | Обновить `.env` → recreate `admin-panel`. |
| `GRAFANA_ADMIN_PASSWORD` | Обновить `.env` → recreate `grafana`. |
| `TELEGRAM_ALERT_CHAT_ID` | Указать chat/user id владельца → recreate `grafana`. |

Живые ключи не вставлять в git, issues и логи чата.

## Инциденты

### Высокий error rate (Grafana: HighErrorRate)

- Порог: failed / (failed + success) **> 5%** на окне **5m**.
- Смотрите логи bot/worker, статус LLM-провайдера и недавние деплои.
- Метрики: `agent_requests_failed_total`, `agent_requests_total`.
- Mitigation: откат image/ревизии, починить ключи провайдера, временно отключить тяжёлые модули через `ENABLED_MODULES`.

### Лаг async-очереди (Grafana: AsyncQueueLag)

- Порог: `agent_async_queue_lag_seconds` **> 300** (5 минут).
- Проверьте здоровье Redis, логи `worker` и статусы `async_tasks` (`queued` / `running`).
- Mitigation: scale/restart `worker`, аккуратно снять зависшие jobs, снизить нагрузку тяжёлыми запросами.

### LLM provider fallback (Grafana: LLMProviderFallback)

- Порог: `sum(increase(agent_llm_fallback_total[10m]))` **> 0**.
- Primary провайдер упал по auth/balance/network; использован secondary.
- Mitigation: восстановить primary ключ/баланс; убедиться, что оба провайдера в `.env`; смотреть `agent_llm_fallback_total`.

### Бот недоступен / конфликты Telegram

- Один polling-инстанс на один bot token.
- Health/monitor loops (если включён `self_diagnostics`) могут уведомить `OWNER_ID` независимо от Grafana.

### Soft-delete / запросы на данные

- По умолчанию `/clear` и admin clear — **soft** (`deleted_at`).
- Soft-delete пользователя в admin: `POST /api/users/{id}/soft_delete`.
- Опциональный gate `SOFT_DELETE_GATE=true` блокирует soft-deleted пользователей в боте (по умолчанию выкл.).
- Hard purge только через retention job или `hard=true` на clear (использовать редко).

## Rollback

1. Найдите последнюю known-good git-ревизию.
2. `git checkout <revision>` (или задеплойте предыдущий image tag).
3. `docker compose up -d --build` для затронутых сервисов.
4. Восстановите SQLite из backup, если schema/data migration несовместима.
5. Smoke: креативный запрос (без поиска) + фактологический (с поиском) в Telegram; проверьте admin `/api/stats` и `/api/usage`.

## Заметки по алертингу

- Grafana contact point использует `$__env{TELEGRAM_BOT_TOKEN}` и `$__env{TELEGRAM_ALERT_CHAT_ID}` — токенов в YAML нет.
- Unified alerting должен быть включён (`GF_UNIFIED_ALERTING_ENABLED=true`).
- In-bot health-алерты на `OWNER_ID` — дополнительный канал, когда Grafana недоступна.
