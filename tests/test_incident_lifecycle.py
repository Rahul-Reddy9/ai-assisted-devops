# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying incident lifecycle state transitions and deduplication."""

import datetime
from devops_ai.core.config import Settings
from devops_ai.incident.detector import IncidentDetector
from devops_ai.incident.models import (
    Alert,
    AlertSource,
    Category,
    Incident,
    IncidentStatus,
    Severity,
    TimelineEntry,
)
from devops_ai.incident.repository import IncidentRepository


def test_incident_lifecycle_transitions(tmp_path):
    """Verify ordered progression through all lifecycle states."""
    repo = IncidentRepository(storage_dir=tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. State: DETECTED
    inc = Incident(
        incident_id="INC-001",
        version=1,
        title="Test Incident",
        category=Category.APPLICATION,
        severity=Severity.HIGH,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
    )
    repo.save(inc)
    assert repo.get("INC-001").status == IncidentStatus.DETECTED

    # 2. State: CORRELATED
    inc.status = IncidentStatus.CORRELATED
    inc.timeline.append(
        TimelineEntry(
            timestamp=now,
            source=AlertSource.LOGS,
            component="app",
            event="Correlated 3 log events",
            severity=Severity.HIGH,
        )
    )
    repo.save(inc)
    assert repo.get("INC-001").status == IncidentStatus.CORRELATED

    # 3. State: INVESTIGATING
    inc.status = IncidentStatus.INVESTIGATING
    repo.save(inc)
    assert repo.get("INC-001").status == IncidentStatus.INVESTIGATING

    # 4. State: ANALYZED
    inc.status = IncidentStatus.ANALYZED
    repo.save(inc)
    assert repo.get("INC-001").status == IncidentStatus.ANALYZED

    # 5. State: RESOLVED
    inc.status = IncidentStatus.RESOLVED
    inc.resolved_at = datetime.datetime.now(datetime.timezone.utc)
    inc.resolution_notes = "Service recovered after dependency restart"
    repo.save(inc)
    resolved_inc = repo.get("INC-001")
    assert resolved_inc.status == IncidentStatus.RESOLVED
    assert resolved_inc.resolved_at is not None

    # 6. State: CLOSED
    inc.status = IncidentStatus.CLOSED
    inc.closed_at = datetime.datetime.now(datetime.timezone.utc)
    repo.save(inc)
    closed_inc = repo.get("INC-001")
    assert closed_inc.status == IncidentStatus.CLOSED
    assert closed_inc.closed_at is not None


def test_alert_deduplication_within_cooldown(tmp_path):
    """Verify duplicate alerts within cooldown window increment recurrence count."""
    repo = IncidentRepository(storage_dir=tmp_path)
    settings = Settings(incident_dedup_cooldown_seconds=300)
    detector = IncidentDetector(repository=repo, settings=settings)

    now = datetime.datetime.now(datetime.timezone.utc)
    alert1 = Alert(
        alert_id="ALT-001",
        source=AlertSource.LOGS,
        timestamp=now,
        component="auth-service",
        message="Token verification failed: timeout",
        severity=Severity.MEDIUM,
        fingerprint="fp-auth-token-timeout",
        recurrence_count=1,
    )

    # First ingestion -> creates new incident
    inc1 = detector.ingest_alert(alert1)
    assert inc1.status == IncidentStatus.DETECTED
    assert len(inc1.alerts) == 1
    assert inc1.alerts[0].recurrence_count == 1

    # Second ingestion within 10 seconds with identical fingerprint
    alert2 = Alert(
        alert_id="ALT-002",
        source=AlertSource.LOGS,
        timestamp=now + datetime.timedelta(seconds=10),
        component="auth-service",
        message="Token verification failed: timeout",
        severity=Severity.MEDIUM,
        fingerprint="fp-auth-token-timeout",
        recurrence_count=1,
    )

    inc2 = detector.ingest_alert(alert2)
    # Must be the exact same incident
    assert inc2.incident_id == inc1.incident_id
    # Recurrence count incremented
    assert inc2.alerts[0].recurrence_count == 2
    # Total stored incidents remains 1
    assert len(repo.list_all()) == 1


def test_reopen_resolved_incident_on_recurring_alert(tmp_path):
    """Verify an incident in RESOLVED state reopens when a new matching alert arrives."""
    repo = IncidentRepository(storage_dir=tmp_path)
    settings = Settings(
        incident_dedup_cooldown_seconds=60,
        incident_reopen_window_seconds=3600,
    )
    detector = IncidentDetector(repository=repo, settings=settings)

    now = datetime.datetime.now(datetime.timezone.utc)
    alert1 = Alert(
        alert_id="ALT-101",
        source=AlertSource.DOCKER,
        timestamp=now,
        component="cache-redis",
        message="Container exited unexpectedly",
        severity=Severity.HIGH,
        fingerprint="fp-redis-exit",
    )

    inc = detector.ingest_alert(alert1)
    assert inc.status == IncidentStatus.DETECTED

    # Operator marks incident as RESOLVED
    inc.status = IncidentStatus.RESOLVED
    inc.resolved_at = now + datetime.timedelta(minutes=5)
    inc.resolution_notes = "Redis container restarted manually"
    repo.save(inc)
    assert repo.get(inc.incident_id).status == IncidentStatus.RESOLVED

    # Recurring alert arrives 15 minutes later (within reopen window of 3600s)
    alert2 = Alert(
        alert_id="ALT-102",
        source=AlertSource.DOCKER,
        timestamp=now + datetime.timedelta(minutes=15),
        component="cache-redis",
        message="Container exited unexpectedly",
        severity=Severity.HIGH,
        fingerprint="fp-redis-exit",
    )

    reopened_inc = detector.ingest_alert(alert2)
    assert reopened_inc.incident_id == inc.incident_id
    assert reopened_inc.status == IncidentStatus.DETECTED
    assert reopened_inc.resolved_at is None
    # Timeline includes reopening event
    assert any("Reopened" in entry.event for entry in reopened_inc.timeline)
