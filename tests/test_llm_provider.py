"""Tests for configurable LLM providers (OpenAI/DeepSeek)."""

from __future__ import annotations

import os

import pytest
from openai import APIStatusError

from agents.llm_config import LLMConfig


def test_get_provider_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    assert LLMConfig.get_provider() == "deepseek"


def test_create_chat_model_uses_selected_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")

    model = LLMConfig.create_chat_model(temperature=0)

    assert model.model_name == "gpt-4o-mini"
    assert model.openai_api_base == "https://api.openai.com/v1"
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "sk-test-openai"


def test_llm_config_works_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError) as error:
        LLMConfig.create_chat_model(temperature=0)

    assert "OPENAI_API_KEY" in str(error.value)


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set",
)
def test_deepseek_real_call_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    if not os.getenv("DEEPSEEK_BASE_URL"):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    model = LLMConfig.create_chat_model(temperature=0)
    try:
        response = model.invoke("1+1=? Ответь только числом.")
    except APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 402 or "Insufficient Balance" in str(exc):
            pytest.skip("DeepSeek balance is insufficient for real-call test (HTTP 402).")
        raise

    assert response.content is not None
    assert "2" in str(response.content)
