# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for LLM provider health monitoring and telemetry sampling."""

from unittest.mock import MagicMock, patch
from devops_ai.observability.provider_monitor import ProviderMonitor, ProviderTelemetry


def test_provider_monitor_probe_healthy():
    mock_provider = MagicMock()
    mock_provider.provider_name = "ollama"
    mock_provider.model_name = "llama3.2:1b"
    mock_provider.host = "http://localhost:11434"
    mock_provider.check_health.return_value = True

    with patch(
        "devops_ai.observability.provider_monitor.LLMRouter.get_provider",
        return_value=mock_provider,
    ):
        monitor = ProviderMonitor()
        telemetry = monitor.probe_provider("ollama")

        assert isinstance(telemetry, ProviderTelemetry)
        assert telemetry.provider == "ollama"
        assert telemetry.is_healthy is True
        assert telemetry.model == "llama3.2:1b"
        assert telemetry.endpoint == "http://localhost:11434"
        assert telemetry.probe_latency_seconds >= 0.0
        assert telemetry.error_message is None


def test_provider_monitor_probe_unhealthy():
    mock_provider = MagicMock()
    mock_provider.provider_name = "gemini"
    mock_provider.model_name = "gemini-2.5-flash"
    mock_provider.host = None
    mock_provider.check_health.return_value = False

    with patch(
        "devops_ai.observability.provider_monitor.LLMRouter.get_provider",
        return_value=mock_provider,
    ):
        monitor = ProviderMonitor()
        telemetry = monitor.probe_provider("gemini")

        assert telemetry.provider == "gemini"
        assert telemetry.is_healthy is False
        assert "remote verification check failed" in (telemetry.error_message or "")


def test_provider_monitor_probe_all():
    mock_provider = MagicMock()
    mock_provider.model_name = "mock-model"
    mock_provider.host = None
    mock_provider.check_health.return_value = True

    with patch(
        "devops_ai.observability.provider_monitor.LLMRouter.get_provider",
        return_value=mock_provider,
    ):
        monitor = ProviderMonitor()
        results = monitor.probe_all()

        assert "ollama" in results
        assert "gemini" in results
        assert results["ollama"].is_healthy is True
        assert results["gemini"].is_healthy is True
