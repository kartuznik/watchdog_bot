"""Centralized LLM provider configuration for agents."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)


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
                "Check ai-agents-lab/.env.example for valid values."
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
                "Set OPENAI_API_KEY in ai-agents-lab/.env."
                if provider == cls.OPENAI_PROVIDER
                else (
                    "Set DEEPSEEK_API_KEY in ai-agents-lab/.env "
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
