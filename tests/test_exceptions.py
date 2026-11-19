# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Domain exceptions for the DevOps AI framework."""

from devops_ai.core.exceptions import (
    ConfigurationError,
    DevOpsAIError,
    LLMProviderError,
    ObservabilityError,
)


def test_base_exception():
    err = DevOpsAIError("Something went wrong", code="CUSTOM_CODE")
    assert str(err) == "[CUSTOM_CODE] Something went wrong"
    assert err.code == "CUSTOM_CODE"


def test_configuration_error():
    err = ConfigurationError("Missing API key")
    assert isinstance(err, DevOpsAIError)
    assert err.code == "CONFIG_ERROR"
    assert "[CONFIG_ERROR] Missing API key" in str(err)


def test_llm_provider_error():
    err = LLMProviderError("Connection refused", provider="ollama")
    assert isinstance(err, DevOpsAIError)
    assert err.provider == "ollama"
    assert err.code == "LLM_PROVIDER_ERROR"
    assert "Provider 'ollama' error: Connection refused" in str(err)


def test_observability_error():
    err = ObservabilityError("Failed to parse log line")
    assert isinstance(err, DevOpsAIError)
    assert err.code == "OBSERVABILITY_ERROR"
