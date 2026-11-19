# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Shared pytest fixtures."""

import pytest
from devops_ai.core.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure settings cache is cleared between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_env(monkeypatch):
    """Provide a clean mock environment for configuration testing."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OLLAMA_HOST", "http://test-ollama:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:1b")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("GEMINI_MODEL", "custom-gemini-model")
