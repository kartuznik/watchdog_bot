# Watchdog — AI Multi-Agent Assistant

Автономный Telegram-ассистент, который **думает, прежде чем ответить**.

В отличие от обычных чат-ботов, Watchdog использует **multi-agent архитектуру** на базе LangGraph: перед ответом запускается цепочка Researcher → Writer → Reviewer, где каждый агент выполняет свою роль. Reviewer может вернуть черновик на доработку, если качество недостаточно — это гарантирует точность и полноту ответов.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m telegram_bot.main
```

Минимум для запуска в `.env`:
- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `ADMIN_PASSWORD` (для веб-админки)

## Features

- 🧠 **Multi-agent pipeline**: `WebSearch -> Researcher -> Writer <-> Reviewer`
- 🔍 **Веб-поиск и TG-парсинг**: Tavily + парсер публичных каналов `t.me/s/...`
- 🧾 **Память диалогов**: SQLite history, восстановление контекста между сообщениями
- 🔖 **Якоря**: быстрые закладки важных фрагментов разговора
- ⚡ **UX без "тишины"**: typing indicator в цикле во время долгих вычислений
- 📊 **Monitoring**: Prometheus + Grafana + метрики запросов/latency/tokens
- 🛠 **Admin panel**: FastAPI dashboard для контроля памяти и якорей

## Architecture

```text
Telegram User
    |
    v
[aiogram handlers]
    |
    +--> [LangGraph]
           START
             |
             v
       [web_search_node] ---> [research_node] ---> [writer_node] ---> [reviewer_node]
                                                             ^               |
                                                             |               |
                                                             +---- feedback --+
                                                                      (max 2 loops)
             |
             +--> LLM Provider (OpenAI / DeepSeek via LLMConfig)
             +--> SQLite Memory + Anchors
             +--> Prometheus Metrics (:8001)
                       |
                       v
                 Prometheus (:9091) ---> Grafana (:3001)
                       |
                       +--> Admin Panel (:8004)
```

## Commands

| Command | Что делает |
|---|---|
| `/start` | Живое приветствие и описание возможностей Watchdog |
| `/research <тема>` | Запуск full multi-agent анализа по теме |
| `/clear` | Очистка истории диалога текущего пользователя |

## LLM Providers

| Provider | Примерная стоимость | Скорость | Качество | Когда использовать |
|---|---:|---|---|---|
| GPT-4o | `$5-10 / 1M` токенов | Средняя/высокая | Очень высокое | Точечный финальный QA |
| GPT-4o-mini | `$0.15 / 1M` токенов | Высокая | Хорошее | **Дефолт для разработки** |
| DeepSeek-V3 | `$0.14 / 1M` токенов | Высокая | Хорошее | Массовые прогоны при пополненном балансе |

## Testing

```bash
pytest tests/ -v
python test_agent_live.py
```

- `test_agent_live.py` делает live-проверку LLM + запускает multi-agent граф.
- В unit-тестах применяются mock-подходы, чтобы CI не зависел от внешних API.

## Deployment (Docker)

```bash
docker compose up -d --build
docker compose ps
```

Сервисы и порты:
- `bot` metrics: `8011 -> 8001`
- `admin-panel`: `8004`
- `prometheus`: `9091`
- `grafana`: `3001`
- `redis`: внутренний сервис

## Troubleshooting

| Проблема | Причина | Решение |
|---|---|---|
| `TELEGRAM_BOT_TOKEN is not set` | Нет токена в `.env` | Добавь `TELEGRAM_BOT_TOKEN` и перезапусти |
| `Conflict: terminated by other getUpdates request` | Два polling-инстанса с одним токеном | Оставь только один запущенный bot process |
| `Insufficient Balance` | Недостаточно средств у LLM-провайдера | Пополни баланс или переключись на `gpt-4o-mini` |
| Нет веб-результатов | Нет `TAVILY_API_KEY` | Добавь ключ Tavily в `.env` |
| Админка отдаёт `401` | Нужна basic auth | Войди `admin` + `ADMIN_PASSWORD` |

---

Если хочешь превратить Watchdog в enterprise-версию, следующий шаг — добавить role-based доступ в админке и отдельный background worker для тяжелых задач поиска/парсинга.
