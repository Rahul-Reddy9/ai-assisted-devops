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
    assert settings.default_llm_provider == "ollama"
    assert settings.enable_llm_fallback is False
    assert settings.llm_timeout_seconds == 60.0
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2:1b"
    assert settings.google_api_key is None
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.anomaly_contamination == 0.1
    assert settings.burst_threshold == 3
    assert settings.burst_window_seconds == 30
    assert settings.rag_max_pdf_pages == 500
    assert settings.rag_embedding_provider == "ollama"
    assert settings.rag_embedding_model == "nomic-embed-text"
    assert settings.rag_embedding_dimension == 768
    assert settings.rag_embedding_batch_size == 32
    assert settings.rag_vector_store == "chroma"
    assert settings.rag_vector_store_path == "./data/vector_store"
    assert settings.rag_retrieval_top_k == 5
    assert settings.rag_similarity_threshold == 0.5


def test_custom_settings_via_env(mock_env):
    settings = Settings()
    assert settings.app_env == "testing"
    assert settings.log_level == "DEBUG"
    assert settings.ollama_host == "http://test-ollama:11434"
    assert settings.google_api_key is not None
    assert settings.google_api_key.get_secret_value() == "test-api-key-12345"
    assert settings.gemini_model == "custom-gemini-model"


def test_secret_str_masking():
    settings = Settings(google_api_key="secret-key-to-mask")
    assert settings.google_api_key is not None
    assert settings.google_api_key.get_secret_value() == "secret-key-to-mask"
    assert "secret-key-to-mask" not in str(settings.google_api_key)
    assert "secret-key-to-mask" not in repr(settings.google_api_key)


def test_invalid_ollama_host():
    with pytest.raises(ValidationError):
        Settings(ollama_host="ftp://invalid-host:11434")


def test_invalid_contamination_range():
    with pytest.raises(ValidationError):
        Settings(anomaly_contamination=0.99)


def test_invalid_timeout():
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=0.5)


def test_invalid_rag_max_pdf_pages():
    with pytest.raises(ValidationError):
        Settings(rag_max_pdf_pages=0)
    with pytest.raises(ValidationError):
        Settings(rag_max_pdf_pages=-10)


def test_invalid_rag_embedding_and_vector_settings():
    """Verify validation boundaries for embedding and vector store settings."""
    with pytest.raises(ValidationError):
        Settings(rag_embedding_provider="unsupported_provider")

    with pytest.raises(ValidationError):
        Settings(rag_vector_store="unsupported_store")

    with pytest.raises(ValidationError):
        Settings(rag_embedding_dimension=0)

    with pytest.raises(ValidationError):
        Settings(rag_embedding_dimension=-5)

    with pytest.raises(ValidationError):
        Settings(rag_embedding_batch_size=0)

    with pytest.raises(ValidationError):
        Settings(rag_retrieval_top_k=0)

    with pytest.raises(ValidationError):
        Settings(rag_similarity_threshold=-0.1)

    with pytest.raises(ValidationError):
        Settings(rag_similarity_threshold=1.5)
