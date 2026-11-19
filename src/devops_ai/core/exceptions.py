# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Domain exceptions for the DevOps AI framework."""


class DevOpsAIError(Exception):
    """Base exception for all DevOps AI errors."""

    def __init__(self, message: str, code: str = "DEVOPS_AI_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigurationError(DevOpsAIError):
    """Raised when configuration values are missing, invalid, or malformed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIG_ERROR")


class LLMProviderError(DevOpsAIError):
    """Raised when an LLM provider fails to generate or connect."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(f"Provider '{provider}' error: {message}", code="LLM_PROVIDER_ERROR")
        self.provider = provider


class ObservabilityError(DevOpsAIError):
    """Raised when log parsing or anomaly detection fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="OBSERVABILITY_ERROR")
