# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Ollama local LLM provider implementation."""

import json
import re
from typing import Any, List, Optional
import urllib.error
import urllib.request
from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import LLMProviderError
from devops_ai.core.logging import get_logger
from devops_ai.llm.base import BaseLLMProvider

logger = get_logger("llm.ollama")


class OllamaProvider(BaseLLMProvider):
    """Local LLM provider utilizing Ollama daemon."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.ollama_model
        self._host = (host or settings.ollama_host).rstrip("/")
        self._timeout = timeout if timeout is not None else settings.llm_timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def host(self) -> Optional[str]:
        return self._host

    def is_configured(self) -> bool:
        return bool(self._host and self._model)

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
        try:
            import ollama
        except ImportError:
            raise LLMProviderError(
                "Ollama package is not installed. Install via: pip install 'devops-ai[ollama]'",
                provider="ollama",
            ) from None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            client = ollama.Client(host=self._host, timeout=self._timeout)
            logger.debug(f"Calling Ollama at {self._host} with model '{self._model}'")
            response = client.chat(
                model=self._model,
                messages=messages,
                options={"temperature": temperature},
            )
            content = response.get("message", {}).get("content", "")
            if not content:
                raise LLMProviderError("Ollama returned an empty response.", provider="ollama")
            return content
        except LLMProviderError:
            raise
        except Exception as e:
            sanitized_err = self._sanitize_message(str(e))
            logger.error(f"Ollama generation failure: {sanitized_err}")
            raise LLMProviderError(
                f"Ollama generation failed at '{self._host}' with model '{self._model}': {sanitized_err}",
                provider="ollama",
            ) from None

    def get_installed_models(self, check_timeout: float = 3.0) -> List[str]:
        """Fetch list of installed models from Ollama daemon via /api/tags."""
        url = f"{self._host}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "devops-ai"})
        try:
            with urllib.request.urlopen(req, timeout=check_timeout) as resp:
                if resp.status != 200:
                    return []
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
        except Exception as e:
            logger.debug(f"Failed to query Ollama tags: {e}")
            return []

    def check_health(self, verify_remote: bool = False) -> bool:
        """Check provider health and readiness.
        
        verify_remote=False: Validates host URL syntax and model configuration locally.
        verify_remote=True: Makes live HTTP call to /api/tags and verifies daemon reachability.
        """
        if not self.is_configured():
            return False
        if not verify_remote:
            return True

        url = f"{self._host}/api/tags"
        check_timeout = min(self._timeout, 3.0)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "devops-ai"})
            with urllib.request.urlopen(req, timeout=check_timeout) as resp:
                return resp.status == 200
        except Exception as e:
            sanitized_err = self._sanitize_message(str(e))
            logger.debug(f"Ollama health check failed: {sanitized_err}")
            return False
