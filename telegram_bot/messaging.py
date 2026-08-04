"""Telegram message helpers: product presentation, chunking, user-facing errors."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from agents.llm_config import LLMConfig
from agents.roles import RoleName, has_required_role
from agents.web_search import SourceItem, normalize_source_items

# Stay under Telegram hard limit (4096) with HTML markup margin.
TELEGRAM_SAFE_CHUNK = 3500
CONTINUATION_SUFFIX = "…\n<i>продолжение ниже</i>"
CONTINUATION_PREFIX = "<i>…продолжение</i>\n"


@dataclass
class OutgoingMessage:
    """One Telegram message ready to send."""

    text: str
    reply_markup: InlineKeyboardMarkup | None = None
    section: str = ""  # draft | research | sources | other


def chunk_text(text: str, max_len: int = TELEGRAM_SAFE_CHUNK) -> list[str]:
    """
    Split text on paragraph/sentence boundaries only.
    Never cuts mid-word; adds continuation markers when needed.
    """
    content = (text or "").strip()
    if not content:
        return []
    if max_len < 128:
        raise ValueError("max_len must be >= 128")
    if len(content) <= max_len:
        return [content]

    # Reserve room for continuation markers on non-final / non-first chunks.
    # Reserve room so markers never push a chunk over max_len.
    body_limit = max_len - len(CONTINUATION_SUFFIX) - len(CONTINUATION_PREFIX) - 8
    pieces: list[str] = []
    remaining = content

    while remaining:
        if len(remaining) <= body_limit:
            pieces.append(remaining)
            break
        window = remaining[:body_limit]
        split_at = _find_boundary(window)
        if split_at <= 0:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = body_limit
        piece = remaining[:split_at].rstrip()
        remaining = remaining[split_at:].lstrip()
        if not piece and remaining:
            piece = remaining[: body_limit - 1] + "…"
            remaining = remaining[body_limit - 1 :].lstrip()
        pieces.append(piece)

    if len(pieces) == 1:
        return pieces

    chunks: list[str] = []
    for index, piece in enumerate(pieces):
        text = piece
        if index < len(pieces) - 1:
            text = text.rstrip() + CONTINUATION_SUFFIX
        if index > 0:
            text = CONTINUATION_PREFIX + text
        chunks.append(text[:max_len])
    return chunks


def _find_boundary(window: str) -> int:
    """Prefer \\n\\n, then sentence end, then single \\n."""
    para = window.rfind("\n\n")
    if para >= len(window) // 4:
        return para
    sentence = -1
    for match in re.finditer(r"[.!?…](?:\s|$)", window):
        sentence = match.end()
    if sentence >= len(window) // 4:
        return sentence
    line = window.rfind("\n")
    if line >= len(window) // 4:
        return line
    return -1


def _escape(text: str) -> str:
    return html.escape(str(text or ""), quote=False)


def _strip_source_appendix(draft: str) -> str:
    """Remove legacy source blocks accidentally left in draft."""
    text = (draft or "").strip()
    patterns = [
        r"\n*###\s*Источники[\s\S]*$",
        r"\n*Источник:\s*\[[^\]]+\][\s\S]*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).rstrip()
    # Drop trailing "Источник: [url]" lines.
    lines = [
        line
        for line in text.splitlines()
        if not re.match(r"^\s*Источник:\s*\[", line, flags=re.IGNORECASE)
    ]
    return "\n".join(lines).strip()


def format_sources_list_html(
    sources: list[SourceItem],
    *,
    limit: int = 5,
    include_domain: bool = False,
) -> str:
    """
    Numbered clickable titles as HTML anchors to the full article URL.
    Domains in parentheses are not used (Telegram would autolink them to site roots).
    Inline buttons still carry the same URLs separately.
    """
    del include_domain  # deprecated: bare domains caused misleading autolinks
    lines: list[str] = []
    for index, item in enumerate(sources[:limit], start=1):
        raw_url = str(item.get("url") or "").strip()
        raw_title = str(item.get("title") or raw_url or "Источник").strip()
        title = html.escape(raw_title, quote=False)
        if not raw_url:
            lines.append(f"{index}. <b>{title}</b>")
            continue
        href = html.escape(raw_url, quote=True)
        lines.append(f'{index}. <a href="{href}"><b>{title}</b></a>')
    return "\n".join(lines)


def _truncate_button_label(label: str, *, limit: int = 64) -> str:
    text = " ".join((label or "").split())
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def build_sources_keyboard(
    sources: list[SourceItem], *, limit: int = 5
) -> InlineKeyboardMarkup | None:
    rows: list[list[InlineKeyboardButton]] = []
    for index, item in enumerate(sources[:limit], start=1):
        title = (item["title"] or item["url"]).strip()
        label = _truncate_button_label(f"{index}. {title}", limit=64)
        rows.append([InlineKeyboardButton(text=label, url=item["url"])])
    if not rows:
        return None
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_tech_footer(result: dict[str, Any]) -> str:
    cost = float(result.get("estimated_cost_usd", 0.0) or 0.0)
    tokens = int(result.get("llm_prompt_tokens", 0) or 0) + int(
        result.get("llm_completion_tokens", 0) or 0
    )
    revisions = int(result.get("revision_count", 0) or 0)
    return (
        f"\n\n<i>Итераций ревью: {revisions} · "
        f"tokens≈{tokens} · cost≈${cost:.6f}</i>"
    )


def build_result_messages(
    result: dict[str, Any],
    *,
    viewer_user_id: int | None = None,
    viewer_role: RoleName | None = None,
    max_len: int = TELEGRAM_SAFE_CHUNK,
) -> list[OutgoingMessage]:
    """
    Product presentation in strict order:
    1) draft  2) research summary  3) sources (+ inline buttons)
    """
    topic = _escape(str(result.get("topic") or "").strip() or "без темы")
    draft = _strip_source_appendix(str(result.get("draft") or "").strip())
    summary = str(result.get("research_summary") or "").strip()
    if not summary:
        # Backward-compatible fallback — never hard-cut mid-word here; chunker handles size.
        summary = str(result.get("research_data") or "").strip()
    sources = normalize_source_items(list(result.get("web_sources") or []))

    if viewer_role is not None:
        show_footer = viewer_role in {"admin", "owner"}
    elif viewer_user_id is not None:
        show_footer = has_required_role(int(viewer_user_id), "admin")
    else:
        show_footer = False

    footer = build_tech_footer(result) if show_footer else ""

    draft_html = (
        f"✅ <b>Готово</b>\n"
        f"<b>Тема:</b> {topic}\n\n"
        f"📝 <b>Ответ</b>\n"
        f"{_escape(draft)}{footer}"
    )
    messages: list[OutgoingMessage] = []
    for chunk in chunk_text(draft_html, max_len=max_len):
        messages.append(OutgoingMessage(text=chunk, section="draft"))

    if summary:
        research_html = f"🔬 <b>Кратко по исследованию</b>\n{_escape(summary)}"
        for chunk in chunk_text(research_html, max_len=max_len):
            messages.append(OutgoingMessage(text=chunk, section="research"))

    if sources:
        sources_html = f"🔗 <b>Источники</b>\n{format_sources_list_html(sources)}"
        keyboard = build_sources_keyboard(sources)
        source_chunks = chunk_text(sources_html, max_len=max_len)
        for index, chunk in enumerate(source_chunks):
            messages.append(
                OutgoingMessage(
                    text=chunk,
                    reply_markup=keyboard if index == 0 else None,
                    section="sources",
                )
            )

    return messages


# Backward-compatible alias used by older tests/call sites.
def build_result_message_parts(
    result: dict[str, Any],
    *,
    research_limit: int = 800,
    max_len: int = TELEGRAM_SAFE_CHUNK,
    viewer_user_id: int | None = None,
    viewer_role: RoleName | None = None,
) -> list[str]:
    del research_limit  # summary is produced in-graph; no mid-word clipping here
    return [
        msg.text
        for msg in build_result_messages(
            result,
            viewer_user_id=viewer_user_id,
            viewer_role=viewer_role,
            max_len=max_len,
        )
    ]


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
        or "invalid api key" in text
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
    # Prefer sentence boundary for memory truncation.
    window = joined[: limit - 1]
    match = None
    for match in re.finditer(r"[.!?…](?:\s|$)", window):
        pass
    if match and match.end() >= (limit // 3):
        return window[: match.end()].rstrip() + "…"
    space = window.rfind(" ")
    if space > limit // 3:
        return window[:space].rstrip() + "…"
    return window.rstrip() + "…"
