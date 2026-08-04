"""Centralized LLM provider configuration for agents."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# Approximate USD prices per 1M tokens: (input, output).
# Used for Prometheus cost estimation — not billing-grade.
_PRICE_PER_1M: dict[tuple[str, str], tuple[float, float]] = {
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("deepseek", "deepseek-chat"): (0.14, 0.28),
    ("deepseek", "deepseek-reasoner"): (0.55, 2.19),
}
_DEFAULT_PRICE_PER_1M = (0.15, 0.60)


class LLMConfig:
    """Resolve provider-specific credentials and create chat models."""

    OPENAI_PROVIDER = "openai"
    DEEPSEEK_PROVIDER = "deepseek"

    DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    DEFAULT_MODEL_BY_PROVIDER = {
        OPENAI_PROVIDER: "gpt-4o-mini",
        DEEPSEEK_PROVIDER: "deepseek-chat",
    }

    @classmethod
    def get_provider(cls) -> str:
        provider = os.getenv("LLM_PROVIDER", cls.OPENAI_PROVIDER).strip().lower()
        if provider not in {cls.OPENAI_PROVIDER, cls.DEEPSEEK_PROVIDER}:
            raise ValueError(
                "Unsupported LLM_PROVIDER. Use 'openai' or 'deepseek'. "
                "See .env.example for valid values."
            )
        return provider

    @classmethod
    def get_api_key(cls) -> str:
        provider = cls.get_provider()
        key_name = (
            "OPENAI_API_KEY"
            if provider == cls.OPENAI_PROVIDER
            else "DEEPSEEK_API_KEY"
        )
        api_key = os.getenv(key_name, "").strip()
        if not api_key:
            provider_hint = (
                "Set OPENAI_API_KEY in .env."
                if provider == cls.OPENAI_PROVIDER
                else (
                    "Set DEEPSEEK_API_KEY in .env "
                    "(get one at https://platform.deepseek.com)."
                )
            )
            raise ValueError(
                f"API key is missing for provider '{provider}'. {provider_hint}"
            )
        return api_key

    @classmethod
    def get_base_url(cls) -> str:
        provider = cls.get_provider()
        if provider == cls.OPENAI_PROVIDER:
            return os.getenv("OPENAI_BASE_URL", cls.DEFAULT_OPENAI_BASE_URL).strip()
        return os.getenv("DEEPSEEK_BASE_URL", cls.DEFAULT_DEEPSEEK_BASE_URL).strip()

    @classmethod
    def get_model_name(cls) -> str:
        model_name = os.getenv("MODEL_NAME", "").strip()
        if model_name:
            return model_name
        provider = cls.get_provider()
        return cls.DEFAULT_MODEL_BY_PROVIDER[provider]

    @classmethod
    def estimate_cost_usd(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> float:
        """Estimate USD cost from token counts for the active (or given) provider/model."""
        resolved_provider = (provider or cls.get_provider()).strip().lower()
        resolved_model = (model_name or cls.get_model_name()).strip().lower()
        input_price, output_price = _PRICE_PER_1M.get(
            (resolved_provider, resolved_model),
            _DEFAULT_PRICE_PER_1M,
        )
        prompt = max(0, int(prompt_tokens))
        completion = max(0, int(completion_tokens))
        return (prompt * input_price + completion * output_price) / 1_000_000.0

    @classmethod
    def create_chat_model(cls, temperature: float = 0) -> ChatOpenAI:
        provider = cls.get_provider()
        model_name = cls.get_model_name()
        base_url = cls.get_base_url()
        api_key = cls.get_api_key()

        logger.info(
            "Creating chat model with provider=%s model=%s base_url=%s",
            provider,
            model_name,
            base_url,
        )
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url,
        )
