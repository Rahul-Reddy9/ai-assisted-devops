# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for Prometheus metrics collection, registry, and caching."""

from devops_ai.observability.metrics import (
    _METRICS_CACHE,
    get_prometheus_exposition,
    record_generation_request,
    record_log_anomaly,
    record_provider_health,
    record_validation_error,
    record_validation_warning,
    update_system_metrics,
)


def test_record_generation_request():
    record_generation_request(
        generator="dockerfile",
        provider="ollama",
        status="success",
        duration_seconds=1.25,
    )
    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_generation_requests_total" in expo
    assert 'generator="dockerfile"' in expo
    assert 'provider="ollama"' in expo
    assert 'status="success"' in expo
    assert "devops_ai_generation_duration_seconds" in expo


def test_record_validation_warning_and_error():
    record_validation_warning(generator="k8s", warning_type="security")
    record_validation_error(generator="k8s", error_code="INVALID_YAML")

    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_validation_warnings_total" in expo
    assert 'warning_type="security"' in expo
    assert "devops_ai_validation_errors_total" in expo
    assert 'error_code="INVALID_YAML"' in expo


def test_record_provider_health():
    record_provider_health(provider="ollama", is_healthy=True, probe_duration=0.045)
    record_provider_health(provider="gemini", is_healthy=False, probe_duration=0.120)

    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_provider_health_status" in expo
    assert 'provider="ollama"} 1.0' in expo
    assert 'provider="gemini"} 0.0' in expo
    assert "devops_ai_provider_probe_duration_seconds" in expo


def test_record_log_anomaly():
    record_log_anomaly("burst")
    record_log_anomaly("ml_isolation_forest")

    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_log_anomalies_detected_total" in expo
    assert 'detection_type="burst"' in expo
    assert 'detection_type="ml_isolation_forest"' in expo


def test_system_metrics_sampling():
    update_system_metrics()
    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_system_cpu_usage_percent" in expo
    assert "devops_ai_system_memory_usage_bytes" in expo


def test_metrics_cache_ttl():
    # Calling get_prometheus_exposition multiple times returns cached bytes quickly
    b1 = _METRICS_CACHE.get_exposition()
    b2 = _METRICS_CACHE.get_exposition()
    assert b1 == b2
