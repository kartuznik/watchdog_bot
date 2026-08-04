"""Prometheus metrics for Watchdog Bot."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

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

agent_router_decisions_total = Counter(
    "agent_router_decisions_total",
    "Router decisions for web search usage.",
    ["decision"],
)

agent_async_queue_lag_seconds = Gauge(
    "agent_async_queue_lag_seconds",
    "Max age in seconds of queued/running async_tasks (bot-side SQLite poll).",
)


def observe_llm_fallback(from_provider: str, to_provider: str) -> None:
    agent_llm_fallback_total.labels(
        from_provider=from_provider.strip().lower() or "unknown",
        to_provider=to_provider.strip().lower() or "unknown",
    ).inc()


def observe_router_decision(decision: str) -> None:
    label = decision.strip().lower()
    if label not in {"search", "no_search"}:
        label = "no_search" if label in {"skip", "false", "0"} else "search"
    agent_router_decisions_total.labels(decision=label).inc()


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


def set_async_queue_lag_seconds(lag_seconds: float) -> None:
    agent_async_queue_lag_seconds.set(max(0.0, float(lag_seconds)))


def refresh_async_queue_lag() -> float:
    """Read SQLite queue lag and publish gauge. Returns the lag value."""
    from agents.database import compute_async_queue_lag_seconds

    lag = compute_async_queue_lag_seconds()
    set_async_queue_lag_seconds(lag)
    return lag
