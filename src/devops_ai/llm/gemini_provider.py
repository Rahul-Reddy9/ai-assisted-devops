# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Google Gemini hosted LLM provider implementation."""

import re
from typing import Any, Optional
from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import ConfigurationError, LLMProviderError
from devops_ai.core.logging import get_logger
from devops_ai.llm.base import BaseLLMProvider

logger = get_logger("llm.gemini")


class GeminiProvider(BaseLLMProvider):
    """Hosted LLM provider utilizing Google GenAI SDK with SecretStr masking."""

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.gemini_model
        self._timeout = timeout if timeout is not None else settings.llm_timeout_seconds

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def host(self) -> Optional[str]:
        return None

    def is_configured(self) -> bool:
        settings = get_settings()
        if not settings.google_api_key:
            return False
        raw_key = settings.google_api_key.get_secret_value()
        return bool(raw_key and raw_key.strip())

    def _get_raw_key(self) -> str:
        settings = get_settings()
        if not settings.google_api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY is not configured. Set GOOGLE_API_KEY in your environment or .env file."
            )
        raw = settings.google_api_key.get_secret_value()
        if not raw or not raw.strip():
            raise ConfigurationError("GOOGLE_API_KEY is empty. Provide a valid Google Gemini API key.")
        return raw.strip()

    def _sanitize_message(self, message: str) -> str:
        """Scrub potential credentials or sensitive tokens from error strings."""
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

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        raw_key = self._get_raw_key()

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise LLMProviderError(
                "Google GenAI package is not installed. Install via: pip install 'devops-ai[gemini]'",
                provider="gemini",
            ) from None

        try:
            timeout_ms = int(self._timeout * 1000)
            client = genai.Client(
                api_key=raw_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_prompt,
            )
            logger.debug(f"Calling Google GenAI with model '{self._model}'")
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            if not response.text:
                raise LLMProviderError("Gemini returned an empty response.", provider="gemini")
            return response.text
        except LLMProviderError:
            raise
        except Exception as e:
            sanitized_err = self._sanitize_message(str(e))
            logger.error(f"Gemini generation failure: {sanitized_err}")
            raise LLMProviderError(f"Gemini API call failed: {sanitized_err}", provider="gemini") from None

    def check_health(self, verify_remote: bool = False) -> bool:
        """Check provider health.
        
        If verify_remote is False: validates local API key configuration presence.
        If verify_remote is True: performs live remote connectivity check against Gemini API.
        """
        if not self.is_configured():
            return False
        if not verify_remote:
            return True

        try:
            from google import genai
            from google.genai import types
            raw_key = self._get_raw_key()
            check_timeout_ms = min(int(self._timeout * 1000), 5000)
            client = genai.Client(
                api_key=raw_key,
                http_options=types.HttpOptions(timeout=check_timeout_ms),
            )
            client.models.get(model=self._model)
            return True
        except Exception as e:
            sanitized_err = self._sanitize_message(str(e))
            logger.debug(f"Gemini remote verification failed: {sanitized_err}")
            return False
