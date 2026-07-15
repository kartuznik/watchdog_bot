# Watchdog — AI Multi-Agent Assistant

Автономный Telegram-ассистент, который **думает, прежде чем ответить**.

В отличие от обычных чат-ботов, Watchdog использует **multi-agent архитектуру** на базе LangGraph: перед ответом запускается цепочка Researcher → Writer → Reviewer, где каждый агент выполняет свою роль.

Researcher собирает факты, Writer превращает их в ответ, Reviewer проверяет качество и при необходимости возвращает черновик на доработку. Пользователь получает не «первую мысль модели», а проверенный результат.

---

## Quick Start

Три команды, чтобы бот ожил:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
python -m telegram_bot.main
```

Минимум в `.env`: `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`, `ADMIN_PASSWORD`.

---

## Features

- 🧠 **Multi-agent reasoning:** `WebSearch -> Researcher -> Writer <-> Reviewer`
- ⚡ **Живой UX:** typing indicator в цикле во время длинных вычислений
- 🔍 **Интернет-контекст:** Tavily web search + parser публичных Telegram-каналов
- 🧾 **Память диалога:** SQLite history, контекст между сообщениями
- 🔖 **Якоря:** быстрые закладки ключевых кусочков диалога
- 🛡 **Модульная архитектура:** feature flags в `config.py` (`ENABLED_MODULES`)
- 📊 **Наблюдаемость:** Prometheus метрики + Grafana dashboards
- 🛠 **Операционка:** FastAPI web-admin панель для памяти и ролей

---

## Architecture

```text
Telegram User
    |
    v
[aiogram handlers]
    |
    +--> [LangGraph pipeline]
           START
             |
             v
       [web_search_node] ---> [research_node] ---> [writer_node] ---> [reviewer_node]
                                                             ^               |
                                                             |               |
                                                             +---- feedback --+
                                                                      (loop cap: 2)
             |
             +--> LLM provider (OpenAI / DeepSeek via LLMConfig)
             +--> Conversation memory + anchors (SQLite)
             +--> Metrics endpoint (:8001)
                       |
                       v
              Prometheus (:9091) ---> Grafana (:3001)
                       |
                       +--> Admin panel (:8004)
```

Потоки данных:
- пользовательский запрос -> `handlers.py` -> `MultiAgentState`;
- результат графа сохраняется в память и отдаётся пользователю;
- метрики и служебные события доступны админке/мониторингу.

---

## Commands

| Команда | Назначение | Доступ |
|---|---|---|
| `/start` | Приветствие и сценарии использования | `user` |
| `/research <тема>` | Глубокий multi-agent анализ | `user` |
| `/clear` | Очистка истории текущего пользователя | `user` |
| `/me` | Показать текущую роль | `user` |
| `/selftest` | Проверка состояния ключевых подсистем | `admin` |
| `/status` | Технический статус и health summary | `admin/owner` |
| `/restart` | Контролируемый перезапуск процесса | `owner` |
| `/setadmin <id>` | Назначить администратора | `owner` |
| `/removeadmin <id>` | Снять права администратора | `owner` |
| `/admins` | Список администраторов | `owner` |

---

## Environment Variables

| Переменная | Обязательность | Что делает |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | **Да** | токен бота от BotFather |
| `OPENAI_API_KEY` | **Да** (для OpenAI) | ключ OpenAI API |
| `LLM_PROVIDER` | Нет | `openai` или `deepseek` |
| `MODEL_NAME` | Нет | модель LLM (по умолчанию `gpt-4o-mini`) |
| `TAVILY_API_KEY` | Нет | веб-поиск для research-узла |
| `ADMIN_PASSWORD` | **Да** для admin-panel | пароль basic auth веб-админки |
| `OWNER_ID` | Нет, но рекомендуется | Telegram user id владельца (owner-role) |
| `REDIS_URL` | Нет | Redis для очередей и worker |
| `AGENT_DB_PATH` | Нет | путь к SQLite базе |
| `METRICS_PORT` | Нет | порт prometheus metrics endpoint |
| `GRAFANA_ADMIN_PASSWORD` | Нет | пароль админа Grafana |

---

## LLM Providers

| Провайдер | Стоимость | Скорость | Качество | Рекомендация |
|---|---:|---|---|---|
| GPT-4o | `$5-10 / 1M` токенов | средняя/высокая | очень высокое | точечный финальный QA |
| GPT-4o-mini | `$0.15 / 1M` токенов | высокая | хорошее | **дефолт для разработки** |
| DeepSeek-V3 | `$0.14 / 1M` токенов | высокая | хорошее | массовые тестовые прогоны |

---

## Testing

```bash
pytest tests/ -v
python test_agent_live.py
```

Практика:
- unit-тесты гоняем в mock-режиме;
- live-check (`test_agent_live.py`) запускаем вручную перед релизом.

---

## Deployment

### Docker Compose

```bash
docker compose up -d --build
docker compose ps
```

Сервисы:
- `bot`
- `worker`
- `redis`
- `admin-panel`
- `prometheus`
- `grafana`

### systemd (для VPS)

```ini
[Unit]
Description=Watchdog Bot
After=network-online.target

[Service]
WorkingDirectory=/opt/watchdog_bot
ExecStart=/opt/watchdog_bot/.venv/bin/python -m telegram_bot.main
Restart=on-failure
User=watchdog
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

---

## Monitoring

- `/selftest` — функциональная проверка подсистем;
- `/status` — технический статус (role-protected);
- Prometheus собирает runtime-метрики;
- Grafana отображает latency/error/token dashboards;
- web-admin показывает диалоги, роли и инструменты ручного восстановления.

---

## Troubleshooting

| Симптом | Причина | Что сделать |
|---|---|---|
| `TELEGRAM_BOT_TOKEN is not set` | переменная не задана | заполни `.env`, перезапусти процесс |
| `Conflict: terminated by other getUpdates request` | два polling-процесса | оставь только один экземпляр бота |
| `Insufficient Balance` | закончился баланс провайдера | пополни баланс или переключись на `gpt-4o-mini` |
| Пустой веб-поиск | нет `TAVILY_API_KEY` | добавь ключ в `.env` |
| `401` на админке | неверный Basic Auth | используй `admin` + `ADMIN_PASSWORD` |
| Нет метрик в Grafana | Prometheus не скрапит bot | проверь `monitoring/prometheus/prometheus.yml` и порты |

---

## Development

Как добавить новую команду без хаоса:

1. Добавь handler в `telegram_bot/handlers.py`.
2. Если команда role-protected — пропиши проверку в `role_check` middleware или декоратор.
3. Добавь/обнови тесты в `tests/`.
4. Обнови таблицу команд в README.
5. Сделай локальный smoke (`pytest` + запуск бота) перед push.

---

Если хочешь следующий уровень, мы можем вынести graph-execution в очередь worker-ов и сделать SLA-наблюдаемость по пользователям/ролям/ошибкам в реальном времени.
