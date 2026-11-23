# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for Phase 4 incident data models and serialization."""

import datetime
import pytest
from pydantic import ValidationError

from devops_ai.incident.models import (
    Alert,
    AlertSource,
    Category,
    CorrelatedEvidence,
    Incident,
    IncidentStatus,
    RemediationAction,
    RiskLevel,
    RootCauseHypothesis,
    Severity,
    TimelineEntry,
)


def test_alert_model_valid():
    """Verify Alert model creation with required and default fields."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = Alert(
        alert_id="ALT-20261123-0001",
        source=AlertSource.LOGS,
        timestamp=now,
        component="payment-service",
        message="Database connection refused on port 5432",
        severity=Severity.HIGH,
        fingerprint="sha256-test-fingerprint",
        recurrence_count=1,
    )
    assert alert.alert_id == "ALT-20261123-0001"
    assert alert.source == AlertSource.LOGS
    assert alert.severity == Severity.HIGH
    assert alert.recurrence_count == 1
    assert alert.raw_data == {}


def test_alert_extra_fields_forbidden():
    """Verify Alert rejects undefined extra attributes."""
    now = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(ValidationError):
        Alert(
            alert_id="ALT-01",
            source=AlertSource.METRICS,
            timestamp=now,
            component="web",
            message="CPU spike",
            severity=Severity.MEDIUM,
            fingerprint="fp1",
            unknown_extra_field="malicious_payload",  # type: ignore[call-arg]
        )


def test_incident_model_round_trip_serialization():
    """Verify Incident model serializes to JSON dict and deserializes cleanly."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = Alert(
        alert_id="ALT-001",
        source=AlertSource.DOCKER,
        timestamp=now,
        component="worker-1",
        message="Container exited with code 137",
        severity=Severity.HIGH,
        fingerprint="fp-docker-137",
    )
    timeline = TimelineEntry(
        timestamp=now,
        source=AlertSource.DOCKER,
        component="worker-1",
        event="Container exited with code 137",
        severity=Severity.HIGH,
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-01": "Exit code 137"},
        container_states=[{"id": "cid-1", "state": "exited", "exit_code": 137}],
    )
    hypothesis = RootCauseHypothesis(
        title="OOM Termination",
        description="Container terminated by OOM killer",
        confidence=0.95,
        tier="tier_1_deterministic",
        evidence_refs=["EVID-01"],
        supporting_evidence=["Container inspect OOMKilled=True"],
        is_confirmed=True,
    )
    remediation = RemediationAction(
        action_id="ACT-01",
        title="Increase memory limit",
        description="Increase container memory limit from 512Mi to 1Gi",
        action_type="immediate_mitigation",
        risk_level=RiskLevel.MODERATE,
        command="kubectl set resources deployment/worker --limits=memory=1Gi",
        verification_step="kubectl get deployment worker -o jsonpath='{.spec.template.spec.containers[0].resources}'",
    )

    incident = Incident(
        incident_id="INC-20261123-0001",
        version=1,
        title="Incident in worker-1: Container exited with code 137",
        category=Category.CONTAINER,
        severity=Severity.HIGH,
        status=IncidentStatus.ANALYZED,
        created_at=now,
        updated_at=now,
        components_affected=["worker-1"],
        alerts=[alert],
        timeline=[timeline],
        evidence=evidence,
        root_cause=hypothesis,
        recommendations=[remediation],
    )

    dumped = incident.model_dump(mode="json")
    assert dumped["incident_id"] == "INC-20261123-0001"
    assert dumped["version"] == 1
    assert dumped["status"] == "ANALYZED"
    assert dumped["root_cause"]["confidence"] == 0.95

    # Round-trip validation
    loaded = Incident.model_validate(dumped)
    assert loaded.incident_id == incident.incident_id
    assert loaded.status == IncidentStatus.ANALYZED
    assert loaded.root_cause is not None
    assert loaded.root_cause.is_confirmed is True
    assert loaded.recommendations[0].risk_level == RiskLevel.MODERATE


def test_hypothesis_confidence_bounds():
    """Verify RootCauseHypothesis confidence is strictly constrained between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        RootCauseHypothesis(
            title="Invalid",
            description="Confidence too high",
            confidence=1.2,  # Invalid
            tier="tier_1_deterministic",
        )

    with pytest.raises(ValidationError):
        RootCauseHypothesis(
            title="Invalid",
            description="Confidence negative",
            confidence=-0.1,  # Invalid
            tier="tier_2_ai",
        )
