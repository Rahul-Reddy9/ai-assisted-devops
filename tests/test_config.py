# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for configuration management."""

import pytest
from pydantic import ValidationError
from devops_ai.core.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2:1b"
    assert settings.google_api_key is None
    assert settings.gemini_model is None
    assert settings.anomaly_contamination == 0.1
    assert settings.burst_threshold == 3
    assert settings.burst_window_seconds == 30


def test_custom_settings_via_env(mock_env):
    settings = Settings()
    assert settings.app_env == "testing"
    assert settings.log_level == "DEBUG"
    assert settings.ollama_host == "http://test-ollama:11434"
    assert settings.google_api_key == "test-api-key-12345"
    assert settings.gemini_model == "custom-gemini-model"


def test_invalid_ollama_host():
    with pytest.raises(ValidationError):
        Settings(ollama_host="ftp://invalid-host:11434")


def test_invalid_contamination_range():
    with pytest.raises(ValidationError):
        Settings(anomaly_contamination=0.99)
