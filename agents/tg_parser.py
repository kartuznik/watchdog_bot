"""Telegram public channel parser adapted from tg_feed_fetcher."""

from __future__ import annotations

import asyncio
import html as html_module
import logging
import re
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
FETCH_TIMEOUT_SEC = 20


def _norm_username(raw: str) -> str:
    cleaned = (raw or "").strip().lstrip("@").lower()
    return re.sub(r"[^a-z0-9_]", "", cleaned)


def extract_telegram_usernames(text: str) -> list[str]:
    raw = text or ""
    candidates = re.findall(r"(?:https?://t\.me/|@)([A-Za-z0-9_]{4,})", raw)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        username = _norm_username(candidate)
        if username and username not in seen:
            seen.add(username)
            out.append(username)
    return out


def _strip_tags(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html_module.unescape(text)


def fetch_public_channel_posts(username: str, *, limit: int = 3) -> list[dict[str, Any]]:
    channel = _norm_username(username)
    if not channel:
        return []

    url = f"https://t.me/s/{channel}"
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SEC) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Failed to fetch TG channel @%s: %s", channel, exc)
        return []

    chunks = body.split("tgme_widget_message_wrap")
    posts: list[dict[str, Any]] = []
    for chunk in chunks[1:]:
        if len(posts) >= max(1, min(limit, 10)):
            break
        m_link = re.search(r'href="(https://t\.me/[^"/]+/\d+)"', chunk)
        if not m_link:
            continue
        post_url = m_link.group(1).split("?")[0]
        m_text = re.search(
            r'class="tgme_widget_message_text[^"]*"[^>]*>([\s\S]*?)</div>',
            chunk,
            flags=re.I,
        )
        raw_text = m_text.group(1) if m_text else ""
        text = " ".join(_strip_tags(raw_text).strip().split())
        title = (text[:80] + "...") if len(text) > 80 else (text or "Пост Telegram")
        posts.append(
            {
                "title": title,
                "url": post_url,
                "content": text[:1200],
                "channel_username": channel,
            }
        )
    return posts


async def fetch_many_channels_async(usernames: list[str], *, per_channel: int = 2) -> list[dict[str, Any]]:
    unique = [_norm_username(u) for u in usernames]
    unique = [u for u in unique if u]
    if not unique:
        return []

    def _collect() -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for channel in unique:
            for post in fetch_public_channel_posts(channel, limit=per_channel):
                normalized_url = str(post.get("url", "")).strip().lower()
                if normalized_url and normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    merged.append(post)
        return merged

    return await asyncio.to_thread(_collect)
