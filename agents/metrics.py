"""Prometheus metrics for Watchdog Bot."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from agents.llm_config import LLMConfig

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

agent_estimated_cost_usd_total = Counter(
    "agent_estimated_cost_usd_total",
    "Estimated LLM spend in USD (approximate catalog prices).",
)

agent_llm_fallback_total = Counter(
    "agent_llm_fallback_total",
    "LLM provider fallback events (OpenAI ↔ DeepSeek).",
    ["from_provider", "to_provider"],
)


def observe_llm_fallback(from_provider: str, to_provider: str) -> None:
    agent_llm_fallback_total.labels(
        from_provider=from_provider.strip().lower() or "unknown",
        to_provider=to_provider.strip().lower() or "unknown",
    ).inc()


def observe_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cost_usd: float | None = None,
) -> None:
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    agent_prompt_tokens_total.inc(prompt)
    agent_completion_tokens_total.inc(completion)
    estimated = (
        float(cost_usd)
        if cost_usd is not None
        else LLMConfig.estimate_cost_usd(prompt, completion)
    )
    if estimated > 0:
        agent_estimated_cost_usd_total.inc(estimated)
