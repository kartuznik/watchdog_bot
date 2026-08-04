"""Telegram message helpers: chunking, result formatting, user-facing errors."""

from __future__ import annotations

from agents.llm_config import LLMConfig
from agents.multi_agent import MultiAgentState, ensure_sources_block

# Stay under Telegram hard limit (4096) with margin for markdown quirks.
TELEGRAM_SAFE_CHUNK = 3800
RESEARCH_SUMMARY_LIMIT = 1500


def chunk_text(text: str, max_len: int = TELEGRAM_SAFE_CHUNK) -> list[str]:
    """Split text into Telegram-safe chunks, preferring paragraph/word boundaries."""
    content = (text or "").strip()
    if not content:
        return []
    if max_len < 64:
        raise ValueError("max_len must be >= 64")
    if len(content) <= max_len:
        return [content]

    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        window = remaining[:max_len]
        split_at = window.rfind("\n\n")
        if split_at < max_len // 3:
            split_at = window.rfind("\n")
        if split_at < max_len // 3:
            split_at = window.rfind(" ")
        if split_at < max_len // 3:
            split_at = max_len
        piece = remaining[:split_at].rstrip()
        if not piece:
            piece = remaining[:max_len]
            split_at = max_len
        chunks.append(piece)
        remaining = remaining[split_at:].lstrip()
    return chunks


def build_result_message_parts(
    result: MultiAgentState,
    *,
    research_limit: int = RESEARCH_SUMMARY_LIMIT,
    max_len: int = TELEGRAM_SAFE_CHUNK,
) -> list[str]:
    """
    Build send-ready messages: draft first (with sources), then shortened research.
    Each returned string is <= max_len.
    """
    topic = str(result.get("topic") or "").strip() or "без темы"
    sources = [str(u).strip() for u in (result.get("web_sources") or []) if str(u).strip()]
    draft = ensure_sources_block(str(result.get("draft") or "").strip(), sources)
    research = str(result.get("research_data") or "").strip()
    if len(research) > research_limit:
        research = research[: research_limit - 1].rstrip() + "…"

    cost = float(result.get("estimated_cost_usd", 0.0) or 0.0)
    tokens = int(result.get("llm_prompt_tokens", 0) or 0) + int(
        result.get("llm_completion_tokens", 0) or 0
    )
    footer = (
        f"\n\n_Итераций ревью: {result.get('revision_count', 0)} · "
        f"tokens≈{tokens} · cost≈${cost:.6f}_"
    )

    sections: list[str] = [
        f"✅ Готово\n**Тема:** {topic}\n\n### Draft\n{draft}{footer}",
    ]
    if research:
        sections.append(f"### Research (кратко)\n{research}")
    if sources and "Источник:" not in draft:
        source_lines = "\n".join(f"Источник: [{url}]" for url in sources[:5])
        sections.append(f"### Источники\n{source_lines}")

    parts: list[str] = []
    for section in sections:
        parts.extend(chunk_text(section, max_len=max_len))
    return parts


def format_user_facing_error(exc: BaseException) -> str:
    """Map exceptions to honest user messages without leaking secrets."""
    text = str(exc).lower()
    name = type(exc).__name__.lower()

    if "message is too long" in text:
        return (
            "Ответ получился слишком длинным для одного сообщения Telegram. "
            "Попробуйте ещё раз — система теперь отправляет ответ частями."
        )

    tavily_auth = (
        "invalidapikey" in name
        or "invalid API key" in text
        or ("tavily" in text and ("unauthorized" in text or "api key" in text))
    )
    if tavily_auth:
        return (
            "Веб-поиск временно недоступен: ключ Tavily невалиден или просрочен. "
            "Могу ответить без live-источников; админу нужно обновить TAVILY_API_KEY."
        )

    if LLMConfig.is_llm_auth_or_balance_error(exc):
        return (
            "Проблема с доступом или балансом LLM-провайдера "
            "(OpenAI/DeepSeek: 401/402/403 или Insufficient Balance). "
            "Проверьте ключ и баланс либо дождитесь автоматического fallback."
        )

    return (
        "Не удалось обработать запрос. Попробуйте упростить формулировку "
        "или повторить позже."
    )


def shorten_for_memory(parts: list[str], *, limit: int = 3500) -> str:
    """Persist a compact assistant memory blob from sent parts."""
    joined = "\n\n".join(parts).strip()
    if len(joined) <= limit:
        return joined
    return joined[: limit - 1].rstrip() + "…"
