# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying Two-Tier incident orchestration, escalation logic, and fail-safe fallback."""

import datetime
from unittest.mock import MagicMock

from devops_ai.incident.ai_engine import Tier2AIEngine
from devops_ai.incident.analyzer import IncidentAnalyzer
from devops_ai.incident.models import (
    Alert,
    AlertSource,
    Category,
    CorrelatedEvidence,
    Incident,
    IncidentStatus,
    RootCauseHypothesis,
    Severity,
)


def _make_base_incident(
    alert_msg: str,
    component: str = "web-service",
    evidence_catalog: dict = None,
    k8s_events: list = None,
    container_states: list = None,
) -> Incident:
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = Alert(
        alert_id="ALT-ORCH-01",
        source=AlertSource.LOGS,
        timestamp=now,
        component=component,
        message=alert_msg,
        severity=Severity.HIGH,
        fingerprint=f"fp-{component}-{alert_msg[:20]}",
    )
    return Incident(
        incident_id="INC-ORCH-001",
        version=1,
        title=f"Incident in {component}",
        category=Category.UNKNOWN,
        severity=Severity.HIGH,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
        alerts=[alert],
        components_affected=[component],
        evidence=CorrelatedEvidence(
            evidence_catalog=evidence_catalog or {"EVID-01": alert_msg},
            k8s_events=k8s_events or [],
            container_states=container_states or [],
        ),
    )


def test_strong_tier_1_result_skips_tier_2():
    """Verify strong deterministic Tier 1 result (confirmed OOM, high confidence >= 0.85) skips Tier 2."""
    incident = _make_base_incident(
        alert_msg="Container exited (137)",
        component="payment-worker",
        container_states=[{"status": "Exited (137)", "exit_code": 137, "oom_killed": True}],
    )

    mock_ai_engine = MagicMock(spec=Tier2AIEngine)
    mock_ai_engine.investigate.return_value = None
    analyzer = IncidentAnalyzer(auto_tier2=True, ai_engine=mock_ai_engine)
    analyzed = analyzer.analyze(incident)

    # Deterministic Tier 1 found confirmed OOM (confidence 0.95, category CONTAINER)
    assert (
        analyzed.root_cause.title
        == "Container Terminated by Linux Kernel Out-Of-Memory Killer (OOMKilled)"
    )
    assert analyzed.root_cause.confidence >= 0.85
    assert analyzed.metadata.get("escalate_to_tier_2") is False

    # AI engine must NOT have been called
    mock_ai_engine.investigate.assert_not_called()
    assert analyzed.metadata["tier_2_attempted"] is False
    assert analyzed.metadata["tier_2_skipped"] is True
    assert "sufficiently high" in analyzed.metadata["tier_2_skip_reason"]

    # Remediation recommendations were generated
    assert len(analyzed.recommendations) >= 1
    assert analyzed.metadata.get("remediation_generated") is True


def test_low_confidence_tier_1_invokes_tier_2():
    """Verify low-confidence deterministic result (< 0.85) triggers Tier 2 AI investigation."""
    incident = _make_base_incident(
        alert_msg="Ambiguous unclassified warning event",
        component="billing-api",
    )

    # Mock AI engine returning high-quality grounded hypothesis
    mock_ai_hypothesis = RootCauseHypothesis(
        title="Third-Party Webhook Latency Cascade",
        description="Downstream partner webhook timed out causing billing worker thread pool exhaustion.",
        confidence=0.82,
        tier="tier_2_ai",
        evidence_refs=["EVID-01"],
        supporting_evidence=["Observed thread exhaustion"],
        is_confirmed=True,
    )
    mock_ai_engine = MagicMock(spec=Tier2AIEngine)
    mock_ai_engine.investigate.return_value = mock_ai_hypothesis

    analyzer = IncidentAnalyzer(auto_tier2=True, ai_engine=mock_ai_engine)
    analyzed = analyzer.analyze(incident)

    # Verify AI engine was called
    mock_ai_engine.investigate.assert_called_once_with(incident)

    # AI hypothesis became primary root cause
    assert analyzed.root_cause.title == "Third-Party Webhook Latency Cascade"
    assert analyzed.root_cause.tier == "tier_2_ai"

    # Previous Tier 1 primary was preserved in alternative hypotheses
    assert len(analyzed.alternative_hypotheses) >= 1
    assert analyzed.alternative_hypotheses[0].tier == "tier_1_deterministic"

    # Timeline entry was appended
    events = [t.event for t in analyzed.timeline]
    assert any("Tier 2 AI RCA completed" in e for e in events)


def test_unsupported_tier_1_hypothesis_invokes_tier_2():
    """Verify hypothesis marked unsupported due to missing evidence triggers Tier 2 escalation."""
    incident = _make_base_incident(
        alert_msg="Failure with ungrounded pattern reference",
        evidence_catalog={},  # empty catalog -> pattern refs will be invalid
    )

    mock_ai_engine = MagicMock(spec=Tier2AIEngine)
    mock_ai_engine.investigate.return_value = None
    analyzer = IncidentAnalyzer(auto_tier2=True, ai_engine=mock_ai_engine)
    analyzed = analyzer.analyze(incident)

    assert analyzed.metadata.get("escalate_to_tier_2") is True
    reasons = analyzed.metadata.get("escalation_reasons", [])
    assert any("threshold" in r.lower() or "unsupported" in r.lower() for r in reasons)
    mock_ai_engine.investigate.assert_called_once()


def test_tier_2_failure_preserves_tier_1_output():
    """Verify that when Tier 2 AI engine fails (returns None), Tier 1 result is fully preserved."""
    incident = _make_base_incident(
        alert_msg="Unexpected internal runtime error",
        component="auth-service",
    )

    mock_ai_engine = MagicMock(spec=Tier2AIEngine)
    mock_ai_engine.investigate.return_value = None  # AI engine failed or timed out

    analyzer = IncidentAnalyzer(auto_tier2=True, ai_engine=mock_ai_engine)
    analyzed = analyzer.analyze(incident)

    # AI was attempted
    mock_ai_engine.investigate.assert_called_once()
    assert analyzed.metadata["tier_2_fallback_used"] is True

    # Tier 1 result preserved as primary root cause!
    assert analyzed.root_cause is not None
    assert analyzed.root_cause.tier == "tier_1_deterministic"

    # Timeline records fallback event
    events = [t.event for t in analyzed.timeline]
    assert any("Tier 2 AI RCA failed" in e or "preserved Tier 1" in e for e in events)


def test_no_unnecessary_repeated_tier_2_requests():
    """Verify an incident that already completed Tier 2 does not invoke AI engine again unless forced."""
    incident = _make_base_incident(alert_msg="Repeated analysis test")
    incident.metadata["tier_2_attempted"] = True
    incident.metadata["escalate_to_tier_2"] = True

    mock_ai_engine = MagicMock(spec=Tier2AIEngine)
    mock_ai_engine.investigate.return_value = None
    analyzer = IncidentAnalyzer(auto_tier2=True, ai_engine=mock_ai_engine)

    # Analyze without force_tier2
    analyzer.analyze(incident, force_tier2=False)
    mock_ai_engine.investigate.assert_not_called()

    # Analyze with force_tier2=True -> invokes AI engine
    analyzer.analyze(incident, force_tier2=True)
    mock_ai_engine.investigate.assert_called_once()


def test_application_and_unknown_categories_follow_escalation_policy():
    """Verify APPLICATION and UNKNOWN category incidents flag escalate_to_tier_2."""
    incident_app = _make_base_incident(alert_msg="Application logic failure")
    incident_app.category = Category.APPLICATION

    analyzer = IncidentAnalyzer(auto_tier2=False)
    analyzed = analyzer.analyze(incident_app)

    assert analyzed.metadata.get("escalate_to_tier_2") is True
    reasons = analyzed.metadata.get("escalation_reasons", [])
    assert any("application" in r.lower() for r in reasons)
    assert analyzed.metadata.get("tier_2_skipped") is True


def test_confirmed_tier_1_root_cause_cannot_be_demoted_by_unconfirmed_ai():
    """Verify confirmed deterministic Tier 1 root cause is preserved over unconfirmed Tier 2 AI hypothesis (LOGIC-01)."""
    incident = _make_base_incident(
        alert_msg="Container exited with code 137",
        component="payment-worker",
        container_states=[{"status": "Exited (137)", "exit_code": 137, "oom_killed": True}],
    )

    # Mock Tier 2 returning an unconfirmed AI speculation with lower confidence
    mock_ai_hypothesis = RootCauseHypothesis(
        title="Application Slow Heap Leak",
        description="Gradual heap allocation growth resulted in memory pressure.",
        confidence=0.74,
        tier="tier_2_ai",
        is_confirmed=False,
        is_unsupported=False,
    )

    mock_ai_engine = MagicMock(spec=Tier2AIEngine)
    mock_ai_engine.investigate.return_value = mock_ai_hypothesis

    analyzer = IncidentAnalyzer(auto_tier2=True, ai_engine=mock_ai_engine)
    # Force Tier 2 to run despite strong Tier 1 result
    analyzed = analyzer.analyze(incident, force_tier2=True)

    # 1. Primary root cause MUST remain the confirmed deterministic Tier 1 finding!
    assert (
        analyzed.root_cause.title
        == "Container Terminated by Linux Kernel Out-Of-Memory Killer (OOMKilled)"
    )
    assert analyzed.root_cause.is_confirmed is True
    assert analyzed.root_cause.confidence >= 0.90
    assert analyzed.root_cause.tier == "tier_1_deterministic"

    # 2. Precedence metadata confirms preservation
    assert analyzed.metadata["root_cause_precedence"] == "tier_1_preserved"
    assert (
        "preserved over Tier 2 hypothesis" in analyzed.metadata["tier_2_override_rejected_reason"]
    )

    # 3. AI hypothesis stored in alternative hypotheses without loss
    assert len(analyzed.alternative_hypotheses) >= 1
    assert analyzed.alternative_hypotheses[0].title == "Application Slow Heap Leak"


def test_confirmed_tier_2_ai_with_higher_confidence_can_override_tier_1():
    """Verify confirmed Tier 2 hypothesis with strictly higher confidence can legitimately override Tier 1 (LOGIC-01)."""
    incident = _make_base_incident(
        alert_msg="High memory utilization warning",
        component="cache-node",
    )
    # Tier 1 has moderate confirmed confidence 0.82
    incident.root_cause = RootCauseHypothesis(
        title="Tier 1 Intermediate Hypothesis",
        description="Preliminary match",
        confidence=0.82,
        tier="tier_1_deterministic",
        is_confirmed=True,
    )

    # Tier 2 provides definitive confirmed hypothesis with higher confidence 0.91
    mock_ai_hypothesis = RootCauseHypothesis(
        title="Redis Eviction Policy Throttling",
        description="AllKeys-LRU eviction thrashing under sustained write volume.",
        confidence=0.91,
        tier="tier_2_ai",
        is_confirmed=True,
    )

    mock_ai_engine = MagicMock(spec=Tier2AIEngine)
    mock_ai_engine.investigate.return_value = mock_ai_hypothesis

    analyzer = IncidentAnalyzer(auto_tier2=True, ai_engine=mock_ai_engine)
    analyzed = analyzer.analyze(incident, force_tier2=True)

    assert analyzed.root_cause.title == "Redis Eviction Policy Throttling"
    assert analyzed.root_cause.confidence == 0.91
    assert analyzed.metadata["root_cause_precedence"] == "tier_2_override"
    assert (
        analyzed.alternative_hypotheses[0].title
        == "High Memory Utilization / Approaching Saturation"
    )
