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
    assert settings.finops_default_currency == "USD"
    assert settings.finops_max_billing_file_size_bytes == 10 * 1024 * 1024
    assert settings.finops_max_records == 100000
    assert settings.finops_idle_cpu_threshold_pct == 2.0
    assert settings.finops_underutilized_cpu_threshold_pct == 15.0
    assert settings.finops_idle_memory_threshold_pct == 15.0
    assert settings.finops_anomaly_z_score_threshold == 2.5
    assert settings.finops_anomaly_min_observations == 7
    assert settings.finops_anomaly_rolling_window_days == 30
    assert settings.finops_forecast_default_horizon_days == 30
    assert settings.finops_forecast_confidence_level == 0.90
    assert settings.finops_forecast_min_observations == 7


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


def test_finops_settings_validation():
    """Verify validation boundaries and custom configuration for FinOps settings."""
    # Valid custom values
    custom = Settings(
        finops_default_currency="EUR",
        finops_max_billing_file_size_bytes=20 * 1024 * 1024,
        finops_max_records=50000,
        finops_idle_cpu_threshold_pct=3.5,
        finops_underutilized_cpu_threshold_pct=20.0,
        finops_idle_memory_threshold_pct=25.0,
    )
    assert custom.finops_default_currency == "EUR"
    assert custom.finops_max_billing_file_size_bytes == 20 * 1024 * 1024
    assert custom.finops_max_records == 50000
    assert custom.finops_idle_cpu_threshold_pct == 3.5
    assert custom.finops_underutilized_cpu_threshold_pct == 20.0
    assert custom.finops_idle_memory_threshold_pct == 25.0

    # Invalid currency code length
    with pytest.raises(ValidationError):
        Settings(finops_default_currency="US")
    with pytest.raises(ValidationError):
        Settings(finops_default_currency="USDD")

    # Invalid file size bounds (< 1024 or > 100MB)
    with pytest.raises(ValidationError):
        Settings(finops_max_billing_file_size_bytes=500)
    with pytest.raises(ValidationError):
        Settings(finops_max_billing_file_size_bytes=101 * 1024 * 1024)

    # Invalid max records (< 1 or > 1,000,000)
    with pytest.raises(ValidationError):
        Settings(finops_max_records=0)
    with pytest.raises(ValidationError):
        Settings(finops_max_records=1000001)

    # Invalid threshold bounds
    with pytest.raises(ValidationError):
        Settings(finops_idle_cpu_threshold_pct=-1.0)
    with pytest.raises(ValidationError):
        Settings(finops_idle_cpu_threshold_pct=25.0)
    with pytest.raises(ValidationError):
        Settings(finops_underutilized_cpu_threshold_pct=60.0)
    with pytest.raises(ValidationError):
        Settings(finops_idle_memory_threshold_pct=60.0)


def test_finops_anomaly_and_forecast_settings_validation():
    """Verify validation boundaries for anomaly and forecast settings."""
    custom = Settings(
        finops_anomaly_z_score_threshold=3.0,
        finops_anomaly_min_observations=14,
        finops_anomaly_rolling_window_days=60,
        finops_forecast_default_horizon_days=45,
        finops_forecast_confidence_level=0.95,
        finops_forecast_min_observations=10,
    )
    assert custom.finops_anomaly_z_score_threshold == 3.0
    assert custom.finops_anomaly_min_observations == 14
    assert custom.finops_anomaly_rolling_window_days == 60
    assert custom.finops_forecast_default_horizon_days == 45
    assert custom.finops_forecast_confidence_level == 0.95
    assert custom.finops_forecast_min_observations == 10

    # Bounds validation
    with pytest.raises(ValidationError):
        Settings(finops_anomaly_z_score_threshold=0.5)
    with pytest.raises(ValidationError):
        Settings(finops_anomaly_z_score_threshold=15.0)

    with pytest.raises(ValidationError):
        Settings(finops_anomaly_min_observations=1)
    with pytest.raises(ValidationError):
        Settings(finops_anomaly_min_observations=100)

    with pytest.raises(ValidationError):
        Settings(finops_anomaly_rolling_window_days=5)
    with pytest.raises(ValidationError):
        Settings(finops_anomaly_rolling_window_days=400)

    with pytest.raises(ValidationError):
        Settings(finops_forecast_default_horizon_days=0)
    with pytest.raises(ValidationError):
        Settings(finops_forecast_default_horizon_days=500)

    with pytest.raises(ValidationError):
        Settings(finops_forecast_confidence_level=0.40)
    with pytest.raises(ValidationError):
        Settings(finops_forecast_confidence_level=1.0)
