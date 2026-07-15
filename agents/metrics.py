"""Prometheus metrics adapted from rag-telegram-bot metrics module."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

agent_requests_total = Counter(
    "agent_requests_total",
    "Total number of successful multi-agent requests.",
)

agent_requests_failed_total = Counter(
    "agent_requests_failed_total",
    "Total number of failed multi-agent requests.",
)

agent_request_duration_seconds = Histogram(
    "agent_request_duration_seconds",
    "Time spent processing one multi-agent request.",
)

agent_prompt_tokens_total = Counter(
    "agent_prompt_tokens_total",
    "Accumulated prompt tokens from LLM responses.",
)

agent_completion_tokens_total = Counter(
    "agent_completion_tokens_total",
    "Accumulated completion tokens from LLM responses.",
)


def observe_token_usage(prompt_tokens: int, completion_tokens: int) -> None:
    agent_prompt_tokens_total.inc(max(0, int(prompt_tokens)))
    agent_completion_tokens_total.inc(max(0, int(completion_tokens)))
