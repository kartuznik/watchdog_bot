# AI Agents Lab

Учебный мини-проект для освоения LangGraph и построения простого агента с маршрутизацией.

## Неделя 1: Simple Agent

### Архитектура графа

```text
START
  |
  v
[classify_node] --(math)--> [math_node] ----\
          |                                  |
          +--(code)--> [code_node] ---------> END
          |                                  |
          +-(general)-> [general_node] -----/
```

- `classify_node` определяет тип запроса: `math`, `code`, `general`.
- Далее conditional edges направляют выполнение в профильный узел.
- Каждый узел обновляет общее состояние `AgentState`.

### Установка и запуск

1. Создай и активируй виртуальное окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Установи зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Подготовь переменные окружения:
   ```bash
   cp .env.example .env
   ```
   Впиши в `.env` значение `OPENAI_API_KEY`.

### Быстрый запуск агента

```bash
python -c "from agents.simple_agent import run_simple_agent; print(run_simple_agent('2+2=?'))"
```

### Тестирование

```bash
pytest tests/ -v
```

### Примеры запросов и ожидаемых ответов

- Запрос: `2+2=?`
  - Классификация: `math`
  - Ответ: `Math result: 4`
- Запрос: `напиши функцию hello world на python`
  - Классификация: `code`
  - Ответ: Python-сниппет с `def hello_world()`
- Запрос: `привет, как дела?`
  - Классификация: `general`
  - Ответ: дружелюбный общий ответ

## Неделя 2: Multi-Agent Loop

### Концепция

На этой неделе строим multi-agent систему с тремя ролями:
- `Researcher` собирает факты по теме.
- `Writer` готовит черновик.
- `Reviewer` проверяет качество и может вернуть задачу на доработку.

### Архитектура цикла

```text
START -> [research_node] -> [writer_node] -> [reviewer_node]
                                          ^         |
                                          |         |
                                          +----(feedback & revision_count < 2)

[reviewer_node] --(no feedback OR revision_count >= 2)--> END
```

### Как избегаем бесконечных циклов

- `Reviewer` повышает `revision_count`, когда находит проблему в черновике.
- Маршрутизация после ревью:
  - если есть `feedback` и `revision_count < 2` -> вернуться в `writer_node`;
  - иначе -> завершить граф (`END`).
- Это гарантирует, что цикл ограничен максимум двумя возвратами.

### Запуск теста недели 2

```bash
pytest tests/test_multi_agent.py -v
```

## Неделя 3: Telegram Integration

### Что интегрировали

- Добавлен Telegram-слой на `aiogram 3` в папке `telegram_bot/`.
- Хендлеры вызывают multi-agent граф асинхронно через `await graph.ainvoke(state)`.
- В state добавлен `user_id`, чтобы изолировать прогоны разных пользователей.

### UX при долгих циклах

Во время выполнения графа (включая возможные итерации Writer <-> Reviewer) бот:
- отправляет промежуточный статус `🔍 Анализирую тему...`;
- отправляет `typing` action (`message.answer_chat_action`) в цикле каждые несколько секунд, чтобы индикатор "печатает..." не пропадал во время долгих итераций.

Это снижает ощущение “зависания” при вычислениях 5-10 секунд.

### Как запустить Telegram-бота

1. В `ai-agents-lab/.env` добавь:
   - `TELEGRAM_BOT_TOKEN=<твой_токен>`;
   - `OPENAI_API_KEY=<ключ>` (опционально для будущих расширений).
2. Из директории `ai-agents-lab` запусти:
   ```bash
   python -m telegram_bot.main
   ```

### Как протестировать `/research`

1. Открой диалог с ботом в Telegram.
2. Отправь команду:
   ```text
   /research ai-агенты в поддержке клиентов
   ```
3. Ожидаемое поведение:
   - сначала придет `🔍 Анализирую тему...` и появится индикатор “печатает...”;
   - затем сообщение обновится финальным результатом (`Research` + `Draft`).

## LLM Providers: OpenAI vs DeepSeek

### Зачем нужен provider abstraction

- Агенты не должны знать о конкретном вендоре LLM.
- В коде используется единый конфиг-слой `agents/llm_config.py`, который выбирает провайдера по `LLM_PROVIDER`.
- Это позволяет быстро переключаться между OpenAI и DeepSeek без изменений логики графов.

### Сравнение провайдеров

| Провайдер | Ориентировочная стоимость | Скорость | Качество | Когда использовать |
|---|---:|---|---|---|
| GPT-4o | ~$5-10 / 1M токенов | Средняя/высокая | Очень высокое | Финальные проверки с максимальным качеством |
| GPT-4o-mini | ~$0.15 / 1M токенов | Высокая | Хорошее | Ежедневная разработка и регресс-тесты |
| DeepSeek-V3 | ~$0.14 / 1M токенов | Высокая | Хорошее | Массовое обучение агентной логики и циклов |

### Как получить DeepSeek API key

1. Зарегистрируйся на [platform.deepseek.com](https://platform.deepseek.com).
2. Создай API key в личном кабинете.
3. Запиши ключ в `DEEPSEEK_API_KEY` в `ai-agents-lab/.env`.

### Примеры `.env` под разные сценарии

#### 1) Только обучение (DeepSeek)

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

#### 2) Только финальные проверки (OpenAI)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

#### 3) Гибрид

```env
# По умолчанию для ежедневной разработки:
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
MODEL_NAME=deepseek-chat

# Перед финальной валидацией вручную переключить:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_key
# MODEL_NAME=gpt-4o-mini
```

### Экономия бюджета ($0.70)

- GPT-4o: ~$0.70 = ~70K-140K токенов (мало для активных multi-agent loops).
- GPT-4o-mini: ~$0.70 = ~4.6M токенов.
- DeepSeek-V3: ~$0.70 = ~5M токенов.

Практическая оценка на обучение: если один агентный прогон в среднем расходует ~1K токенов, то:
- GPT-4o: ~70-140 прогонов;
- GPT-4o-mini: ~4600 прогонов;
- DeepSeek-V3: ~5000 прогонов.

Рекомендация: DeepSeek для ежедневных итераций, GPT-4o-mini для финальных проверок качества.

## Budget & Cost Optimization

Для текущего бюджета безопасный дефолт для разработки — `gpt-4o-mini`.

| Модель | Примерная цена за 1M токенов | Что дает $0.70 | Рекомендация |
|---|---:|---:|---|
| GPT-4o | ~$5-10 | ~70K-140K токенов | Использовать точечно для финального QA |
| GPT-4o-mini | ~$0.15 | ~4.6M токенов | **Дефолт для разработки и обучения** |
| DeepSeek-V3 | ~$0.14 | ~5M токенов | Опционально, если аккаунт пополнен |

Практический режим:
- ежедневная разработка и тесты: `LLM_PROVIDER=openai`, `MODEL_NAME=gpt-4o-mini`;
- тяжелые/дорогие real-call тесты: запускать вручную;
- unit-тесты: оставлять mock-подход без сетевых вызовов.

## Production Architecture

Ниже production-схема, собранная адаптацией проверенных модулей из других ботов:

```text
Telegram User
    |
    v
[aiogram handlers] ---> [LangGraph: WebSearch -> Researcher -> Writer <-> Reviewer]
    |                                   |                 |
    |                                   |                 +--> LLM Provider (OpenAI / DeepSeek via LLMConfig)
    |                                   |
    |                                   +--> Tavily Web Search + TG channel parser
    |
    +--> SQLite Memory + Anchors (conversation history, bookmarks)
    |
    +--> Prometheus Metrics (:8001)
              |
              v
       Prometheus (:9091) ---> Grafana (:3001)
              |
              +--> FastAPI Admin Panel (:8004)
```

### Какие модули адаптированы

- **Real LLM + orchestration:** `agents/multi_agent.py` (узлы Researcher/Writer/Reviewer + loop) через `agents/llm_config.py`.
- **Web search:** `agents/web_search.py` + `agents/tg_parser.py` (адаптация идеи `search_with_tavily` и парсинга `t.me/s/...`).
- **Memory + anchors + storage:** `agents/memory.py`, `agents/anchors.py`, `agents/database.py`.
- **Admin panel:** `admin_panel/app.py` (FastAPI, endpoints для диалогов/памяти/якорей).
- **Monitoring:** `agents/metrics.py` + `monitoring/prometheus` + `monitoring/grafana`.

### Deployment (Docker Compose)

1. Заполни `ai-agents-lab/.env` (минимум: `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`, `ADMIN_PASSWORD`).
2. Запусти стек:
   ```bash
   docker compose up -d --build
   ```
3. Проверь состояние:
   ```bash
   docker compose ps
   ```
4. Доступ к сервисам:
   - Admin panel: [http://localhost:8004](http://localhost:8004)
   - Prometheus: [http://localhost:9091](http://localhost:9091)
   - Grafana: [http://localhost:3001](http://localhost:3001)

### Operational Notes

- Порты изолированы от основного RAG-стека:
  - admin-panel: `8004`
  - prometheus: `9091`
  - grafana: `3001`
- Для веб-поиска нужен `TAVILY_API_KEY`.
- Для стабильного polling должен работать только **один** инстанс бота с тем же `TELEGRAM_BOT_TOKEN`.
