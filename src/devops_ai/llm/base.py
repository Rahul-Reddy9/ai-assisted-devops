# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Abstract base class for all LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLMProvider(ABC):
    """Abstract base class establishing the contract for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        """Generate text completion from prompt."""

    @abstractmethod
    def check_health(self, verify_remote: bool = False) -> bool:
        """Check provider health and readiness.
        
        If verify_remote is False, checks local configuration without network calls.
        If verify_remote is True, performs a live network probe.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique canonical name of the provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model name."""

    @property
    @abstractmethod
    def host(self) -> Optional[str]:
        """Return the configured host endpoint URL, if applicable."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return whether provider has valid credentials / configuration."""
