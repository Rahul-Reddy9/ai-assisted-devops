# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""LLM Router and factory for selecting and managing providers."""

from dataclasses import dataclass
import re
from typing import Any, Optional
from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import ConfigurationError, LLMProviderError
from devops_ai.core.logging import get_logger
from devops_ai.llm.base import BaseLLMProvider
from devops_ai.llm.gemini_provider import GeminiProvider
from devops_ai.llm.ollama_provider import OllamaProvider

logger = get_logger("llm.router")


@dataclass
class GenerationResult:
    """Structured result returned by LLM generation operations."""

    content: str
    provider: str
    model: str
    fallback_used: bool = False


class LLMRouter:
    """Manages provider resolution, initialization, and fallback."""

    @staticmethod
    def _sanitize_secret(message: str) -> str:
        """Best-effort redaction of API keys or tokens from exception strings."""
        try:
            settings = get_settings()
            sanitized = message
            if settings.google_api_key:
                raw = settings.google_api_key.get_secret_value()
                if raw and raw in sanitized:
                    sanitized = sanitized.replace(raw, "[REDACTED_API_KEY]")
            sanitized = re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", sanitized)
            return sanitized
        except Exception:
            return re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", message)

    @staticmethod
    def get_provider(
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> BaseLLMProvider:
        settings = get_settings()
        selected = (provider_name or settings.default_llm_provider).lower().strip()

        if selected == "ollama":
            return OllamaProvider(model=model, host=host, timeout=timeout)
        elif selected == "gemini":
            return GeminiProvider(model=model, timeout=timeout)
        else:
            raise ConfigurationError(
                f"Unsupported LLM provider: '{selected}'. Supported providers: 'ollama', 'gemini'."
            )

    @staticmethod
    def generate_with_fallback(
        prompt: str,
        system_prompt: Optional[str] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        allow_fallback: Optional[bool] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> GenerationResult:
        settings = get_settings()
        primary_name = (provider_name or settings.default_llm_provider).lower().strip()
        do_fallback = allow_fallback if allow_fallback is not None else settings.enable_llm_fallback

        primary_provider = LLMRouter.get_provider(
            provider_name=primary_name,
            model=model,
            host=host,
            timeout=timeout,
        )

        try:
            content = primary_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
            return GenerationResult(
                content=content,
                provider=primary_provider.provider_name,
                model=primary_provider.model_name,
                fallback_used=False,
            )
        except Exception as primary_err:
            sanitized_primary_msg = LLMRouter._sanitize_secret(str(primary_err))
            if not do_fallback:
                logger.error(f"Primary provider '{primary_name}' failed: {sanitized_primary_msg}")
                raise LLMProviderError(sanitized_primary_msg, provider=primary_name) from None

            alternate_name = "gemini" if primary_name == "ollama" else "ollama"
            logger.warning(
                f"Primary provider '{primary_name}' failed ({sanitized_primary_msg}). "
                f"Attempting fallback to '{alternate_name}'..."
            )

            try:
                # Fallback provider uses its standard configured model/host to prevent passing
                # Ollama-specific hosts or models to Gemini (or vice versa).
                alternate_provider = LLMRouter.get_provider(
                    provider_name=alternate_name,
                    timeout=timeout,
                )
                content = alternate_provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs,
                )
                logger.info(f"Fallback to '{alternate_name}' succeeded.")
                return GenerationResult(
                    content=content,
                    provider=alternate_provider.provider_name,
                    model=alternate_provider.model_name,
                    fallback_used=True,
                )
            except Exception as alt_err:
                sanitized_alt_msg = LLMRouter._sanitize_secret(str(alt_err))
                raise LLMProviderError(
                    f"Primary provider '{primary_name}' failed ({sanitized_primary_msg}) "
                    f"and fallback provider '{alternate_name}' also failed ({sanitized_alt_msg}).",
                    provider="router",
                ) from None


def get_llm_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    host: Optional[str] = None,
    timeout: Optional[float] = None,
) -> BaseLLMProvider:
    return LLMRouter.get_provider(
        provider_name=provider_name,
        model=model,
        host=host,
        timeout=timeout,
    )
