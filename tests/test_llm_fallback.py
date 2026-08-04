"""Tests for OpenAI ↔ DeepSeek auth/billing fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.llm_config import LLMConfig
from agents.metrics import agent_llm_fallback_total, observe_llm_fallback
from agents.multi_agent import _invoke_llm


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeLLM:
    def __init__(self, *, fail: Exception | None = None, content: str = "ok") -> None:
        self.fail = fail
        self.content = content
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(
            content=self.content,
            usage_metadata={"input_tokens": 3, "output_tokens": 2},
            response_metadata={},
        )


def test_is_llm_auth_or_balance_error_detects_402() -> None:
    exc = _FakeAPIStatusError("Error code: 402 - Insufficient Balance", status_code=402)
    assert LLMConfig.is_llm_auth_or_balance_error(exc) is True


def test_is_llm_auth_or_balance_error_ignores_timeout() -> None:
    assert LLMConfig.is_llm_auth_or_balance_error(TimeoutError("deadline")) is False


@pytest.mark.asyncio
async def test_invoke_llm_falls_back_on_402(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")

    primary = _FakeLLM(
        fail=_FakeAPIStatusError("Insufficient Balance", status_code=402),
    )
    secondary = _FakeLLM(content="fallback-answer")
    created: list[str] = []

    def _fake_create(temperature: float = 0, *, provider: str | None = None):
        resolved = provider or LLMConfig.get_provider()
        created.append(resolved)
        if resolved == "openai":
            return primary
        return secondary

    monkeypatch.setattr(LLMConfig, "create_chat_model", staticmethod(_fake_create))

    before = agent_llm_fallback_total.labels(
        from_provider="openai",
        to_provider="deepseek",
    )._value.get()

    result = await _invoke_llm(
        system_prompt="sys",
        user_prompt="user",
        temperature=0,
    )

    assert result is not None
    text, prompt_tokens, completion_tokens = result
    assert text == "fallback-answer"
    assert prompt_tokens == 3
    assert completion_tokens == 2
    assert created == ["openai", "deepseek"]
    assert primary.calls == 1
    assert secondary.calls == 1

    after = agent_llm_fallback_total.labels(
        from_provider="openai",
        to_provider="deepseek",
    )._value.get()
    assert after == before + 1


def test_observe_llm_fallback_increments_counter() -> None:
    before = agent_llm_fallback_total.labels(
        from_provider="deepseek",
        to_provider="openai",
    )._value.get()
    observe_llm_fallback("deepseek", "openai")
    after = agent_llm_fallback_total.labels(
        from_provider="deepseek",
        to_provider="openai",
    )._value.get()
    assert after == before + 1
