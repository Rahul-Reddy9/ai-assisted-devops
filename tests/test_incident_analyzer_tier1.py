# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying Tier 1 deterministic RCA, hypothesis generation, and Tier 2 escalation rules."""

import datetime
from devops_ai.incident.analyzer import IncidentAnalyzer
from devops_ai.incident.models import (
    Alert,
    AlertSource,
    Category,
    CorrelatedEvidence,
    Incident,
    IncidentStatus,
    Severity,
)


def _make_alert(
    alert_id: str,
    source: AlertSource,
    component: str,
    message: str,
    severity: Severity = Severity.MEDIUM,
    raw: dict = None,
) -> Alert:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Alert(
        alert_id=alert_id,
        source=source,
        timestamp=now,
        component=component,
        message=message,
        severity=severity,
        fingerprint=f"fp-{alert_id}",
        recurrence_count=1,
        raw_data=raw or {},
    )


def test_analyzer_single_high_confidence_hypothesis():
    """Verify single clear failure (CrashLoopBackOff) generates primary hypothesis with high confidence."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = _make_alert(
        "ALT-1",
        AlertSource.KUBERNETES,
        "payment-api",
        "Pod entering CrashLoopBackOff",
        Severity.HIGH,
    )
    inc = Incident(
        incident_id="INC-RCA-001",
        version=1,
        title="Payment API CrashLoopBackOff",
        category=Category.UNKNOWN,
        severity=Severity.HIGH,
        status=IncidentStatus.CORRELATED,
        created_at=now,
        updated_at=now,
        alerts=[alert],
        components_affected=["payment-api"],
        evidence=CorrelatedEvidence(
            evidence_catalog={
                "EVID-K8S-01": "CrashLoopBackOff: back-off 40s restarting failed container"
            },
            k8s_events=[{"reason": "CrashLoopBackOff", "restarts": 4}],
            log_snippets=["Fatal error: configuration file not found: /etc/app.yaml"],
        ),
    )

    analyzer = IncidentAnalyzer()
    analyzed = analyzer.analyze(inc)

    assert analyzed.status == IncidentStatus.ANALYZED
    assert analyzed.root_cause is not None
    assert analyzed.root_cause.title == "Kubernetes Pod Entering CrashLoopBackOff"
    assert analyzed.root_cause.tier == "tier_1_deterministic"
    assert analyzed.root_cause.confidence >= 0.85
    assert analyzed.root_cause.is_confirmed is True
    assert "EVID-K8S-01" in analyzed.root_cause.evidence_refs
    assert len(analyzed.root_cause.supporting_evidence) > 0


def test_analyzer_multiple_competing_hypotheses_and_tier2_escalation():
    """Verify competing hypotheses with close confidence triggers Tier 2 escalation."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert1 = _make_alert(
        "ALT-1", AlertSource.DOCKER, "worker", "Container exited (137)", Severity.HIGH
    )
    alert2 = _make_alert(
        "ALT-2", AlertSource.LOGS, "worker", "Upstream request timeout after 30s", Severity.MEDIUM
    )

    inc = Incident(
        incident_id="INC-COMPETING",
        version=1,
        title="Worker Termination and Timeout",
        category=Category.UNKNOWN,
        severity=Severity.HIGH,
        status=IncidentStatus.CORRELATED,
        created_at=now,
        updated_at=now,
        alerts=[alert1, alert2],
        components_affected=["worker"],
        evidence=CorrelatedEvidence(
            evidence_catalog={
                "EVID-DOCKER-01": "Container exited (137)",
                "EVID-LOG-01": "504 Gateway Time-out: upstream request timeout after 30000ms",
            },
            container_states=[{"status": "Exited (137)", "exit_code": 137, "oom_killed": False}],
        ),
    )

    analyzer = IncidentAnalyzer()
    analyzed = analyzer.analyze(inc)

    assert analyzed.root_cause is not None
    assert len(analyzed.alternative_hypotheses) >= 1
    # Check competing hypotheses delta: h1 is ambiguous SIGKILL (0.51), h2 is upstream timeout (0.6375)
    # Both are within 0.15 delta or confidence < 0.85, so Tier 2 escalation must be flagged!
    assert analyzed.metadata.get("escalate_to_tier_2") is True
    reasons = analyzed.metadata.get("escalation_reasons", [])
    assert any("threshold" in r or "competing" in r or "margin" in r for r in reasons)


def test_analyzer_evidence_id_validation_filters_hallucinated_refs():
    """Verify invalid or non-existent evidence IDs are strictly filtered out from evidence_refs."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = _make_alert("ALT-OOM", AlertSource.DOCKER, "svc", "Container exited 137", Severity.HIGH)

    inc = Incident(
        incident_id="INC-PROVENANCE",
        version=1,
        title="Provenance Test",
        category=Category.UNKNOWN,
        severity=Severity.HIGH,
        status=IncidentStatus.CORRELATED,
        created_at=now,
        updated_at=now,
        alerts=[alert],
        components_affected=["svc"],
        evidence=CorrelatedEvidence(
            evidence_catalog={"EVID-VALID-01": "Container exited 137"},
            container_states=[{"status": "Exited (137)", "oom_killed": True}],
        ),
    )

    analyzer = IncidentAnalyzer()
    analyzed = analyzer.analyze(inc)

    # Every item in evidence_refs MUST exist in valid_catalog_keys or alerts
    valid_keys = set(inc.evidence.evidence_catalog.keys())
    valid_keys.add(f"EVID-ALERT-{alert.alert_id}")
    for ref in analyzed.root_cause.evidence_refs:
        assert ref in valid_keys


def test_analyzer_contradictory_evidence_reduces_confidence():
    """Verify contradictory health probe heavily penalizes confidence score."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = _make_alert(
        "ALT-DB", AlertSource.LOGS, "backend", "connection refused to database", Severity.HIGH
    )
    probe_ok = _make_alert(
        "ALT-OK",
        AlertSource.HEALTH_PROBE,
        "backend",
        "probe ok",
        Severity.INFO,
        raw={"status_code": 200},
    )

    inc = Incident(
        incident_id="INC-CONTRA",
        version=1,
        title="Contradictory Telemetry",
        category=Category.APPLICATION,
        severity=Severity.HIGH,
        status=IncidentStatus.CORRELATED,
        created_at=now,
        updated_at=now,
        alerts=[alert, probe_ok],
        components_affected=["backend"],
        evidence=CorrelatedEvidence(
            evidence_catalog={"EVID-LOG-01": "connection refused to database:5432"},
        ),
    )

    analyzer = IncidentAnalyzer()
    analyzed = analyzer.analyze(inc)

    assert analyzed.root_cause is not None
    assert analyzed.root_cause.confidence < 0.50
    assert len(analyzed.root_cause.contradicting_evidence) > 0


def test_analyzer_unknown_fallback_when_no_pattern_matches():
    """Verify unclassified telemetry falls back gracefully to Unknown hypothesis."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = _make_alert(
        "ALT-WEIRD",
        AlertSource.LOGS,
        "custom-daemon",
        "Unexpected state machine tick event #4812",
        Severity.LOW,
    )

    inc = Incident(
        incident_id="INC-UNKNOWN",
        version=1,
        title="Unknown State Tick",
        category=Category.UNKNOWN,
        severity=Severity.LOW,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
        alerts=[alert],
        components_affected=["custom-daemon"],
        evidence=CorrelatedEvidence(
            evidence_catalog={"EVID-LOG-01": "Unexpected state machine tick event #4812"},
        ),
    )

    analyzer = IncidentAnalyzer()
    analyzed = analyzer.analyze(inc)

    assert analyzed.root_cause is not None
    assert "Unknown" in analyzed.root_cause.title
    assert analyzed.root_cause.confidence <= 0.40
    assert analyzed.root_cause.is_confirmed is False
    assert analyzed.metadata.get("escalate_to_tier_2") is True


def test_analyzer_offline_execution_without_llm():
    """Verify Tier 1 analyzer executes completely offline in milliseconds without network or LLM."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = _make_alert(
        "ALT-PULL", AlertSource.KUBERNETES, "frontend", "ImagePullBackOff", Severity.HIGH
    )

    inc = Incident(
        incident_id="INC-OFFLINE",
        version=1,
        title="ImagePullBackOff",
        category=Category.CONTAINER,
        severity=Severity.HIGH,
        status=IncidentStatus.CORRELATED,
        created_at=now,
        updated_at=now,
        alerts=[alert],
        components_affected=["frontend"],
        evidence=CorrelatedEvidence(
            evidence_catalog={
                "EVID-K8S-01": "Failed to pull image registry/frontend:v2: ImagePullBackOff"
            },
            k8s_events=[{"reason": "ImagePullBackOff"}],
        ),
    )

    start = datetime.datetime.now()
    analyzer = IncidentAnalyzer()
    analyzed = analyzer.analyze(inc)
    duration = (datetime.datetime.now() - start).total_seconds()

    # Must finish in under 200 milliseconds (pure offline CPU computation)
    assert duration < 0.20
    assert analyzed.root_cause is not None
    assert analyzed.root_cause.title == "Kubernetes ImagePullBackOff or Image Pull Failure"
