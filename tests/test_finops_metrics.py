# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for FinOps Prometheus telemetry metrics and recording helpers."""

from devops_ai.observability.metrics import (
    get_prometheus_exposition,
    record_finops_agent_execution,
    record_finops_anomaly_identified,
    record_finops_operation,
    record_finops_recommendation,
)


def test_record_finops_operation():
    """Verify operation counter and duration histogram recording with bounded labels."""
    record_finops_operation("analyze", "success", duration_seconds=0.15)
    record_finops_operation("optimize", "failure", duration_seconds=0.45)
    record_finops_operation("advise", "success", duration_seconds=1.20)
    # Test invalid / unbound labels fallback
    record_finops_operation("invalid_op", "unsupported_status", duration_seconds=0.10)

    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_finops_operations_total" in expo
    assert 'operation="analyze",status="success"' in expo
    assert 'operation="optimize",status="failure"' in expo
    assert 'operation="advise",status="success"' in expo
    assert 'operation="unknown",status="unknown"' in expo
    assert "devops_ai_finops_operation_duration_seconds_count" in expo


def test_record_finops_agent_execution():
    """Verify advisory agent execution counter and duration histogram."""
    record_finops_agent_execution("success", "ollama", duration_seconds=2.5)
    record_finops_agent_execution("fallback", "offline", duration_seconds=0.05)
    record_finops_agent_execution("failure", "custom_unknown_prov", duration_seconds=None)

    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_finops_agent_executions_total" in expo
    assert 'provider="ollama",status="success"' in expo
    assert 'provider="offline",status="fallback"' in expo
    assert 'provider="unknown",status="failure"' in expo
    assert "devops_ai_finops_agent_duration_seconds_bucket" in expo


def test_record_finops_recommendation_and_savings():
    """Verify recommendation counting by category, confidence tier, and savings accumulation."""
    # High confidence (>= 0.8) guaranteed savings
    record_finops_recommendation(
        category="compute",
        confidence_score=0.95,
        savings_type="guaranteed",
        monthly_savings=250.0,
    )
    # Medium confidence (0.5 <= score < 0.8) estimated savings
    record_finops_recommendation(
        category="storage",
        confidence_score=0.70,
        savings_type="estimated",
        monthly_savings=120.0,
    )
    # Low confidence (< 0.5) with zero savings
    record_finops_recommendation(
        category="database",
        confidence_score=0.35,
        savings_type="estimated",
        monthly_savings=0.0,
    )
    # Unbound category fallback to "general"
    record_finops_recommendation(
        category="unknown_category_xyz",
        confidence_score=0.85,
        savings_type="invalid_savings_type",
        monthly_savings=50.0,
    )

    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_finops_recommendations_total" in expo
    assert 'category="compute",confidence_tier="high",savings_type="guaranteed"' in expo
    assert 'category="storage",confidence_tier="medium",savings_type="estimated"' in expo
    assert 'category="database",confidence_tier="low",savings_type="estimated"' in expo
    assert 'category="general",confidence_tier="high",savings_type="estimated"' in expo

    assert "devops_ai_finops_savings_identified_dollars_total" in expo
    assert 'category="compute",savings_type="guaranteed"' in expo
    assert 'category="storage",savings_type="estimated"' in expo


def test_record_finops_anomaly_identified():
    """Verify anomaly detection counter by severity level."""
    record_finops_anomaly_identified("CRITICAL")
    record_finops_anomaly_identified("HIGH")
    record_finops_anomaly_identified("low")
    record_finops_anomaly_identified("invalid_sev")  # fallback to MEDIUM

    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    assert "devops_ai_finops_anomalies_identified_total" in expo
    assert 'severity="CRITICAL"' in expo
    assert 'severity="HIGH"' in expo
    assert 'severity="LOW"' in expo
    assert 'severity="MEDIUM"' in expo


def test_cardinality_and_no_data_leakage():
    """Ensure sensitive data (account IDs, resource IDs, customer names) are never in labels."""
    expo = get_prometheus_exposition(force_refresh=True).decode("utf-8")
    # Assert no sensitive patterns exist in label values
    assert "i-0" not in expo
    assert "vol-0" not in expo
    assert "arn:aws" not in expo
    assert "account_" not in expo
