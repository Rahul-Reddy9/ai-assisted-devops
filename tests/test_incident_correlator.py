# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying multi-source incident correlation and sliding window behavior."""

import datetime
from devops_ai.incident.correlator import IncidentCorrelator
from devops_ai.incident.models import Alert, AlertSource, IncidentStatus, Severity


def _create_alert(
    alert_id: str,
    component: str,
    message: str,
    timestamp: datetime.datetime,
    source: AlertSource = AlertSource.LOGS,
    severity: Severity = Severity.MEDIUM,
    raw_data: dict = None,
) -> Alert:
    """Helper creating a test alert."""
    return Alert(
        alert_id=alert_id,
        source=source,
        timestamp=timestamp,
        component=component,
        message=message,
        severity=severity,
        fingerprint=f"fp-{source.value}-{component}-{message[:20]}",
        recurrence_count=1,
        raw_data=raw_data or {},
    )


def test_events_inside_sliding_window_are_correlated():
    """Verify alerts within 300s with matching components are correlated into a single incident."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(seconds=120)

    alert1 = _create_alert("ALT-001", "web-app", "Error burst detected", t0, AlertSource.LOGS)
    alert2 = _create_alert(
        "ALT-002", "web-app", "Health check failed 503", t1, AlertSource.HEALTH_PROBE
    )

    correlator = IncidentCorrelator(window_seconds=300)
    incidents = correlator.correlate_batch([alert1, alert2])

    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.status == IncidentStatus.CORRELATED
    assert len(inc.alerts) == 2
    assert "web-app" in inc.components_affected
    assert len(inc.evidence.evidence_catalog) >= 2


def test_events_outside_sliding_window_are_not_correlated():
    """Verify alerts occurring beyond the sliding window (350s > 300s) form separate incidents."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(seconds=350)

    alert1 = _create_alert("ALT-001", "api-server", "Request timeout", t0)
    alert2 = _create_alert("ALT-002", "api-server", "Health check failed", t1)

    correlator = IncidentCorrelator(window_seconds=300)
    incidents = correlator.correlate_batch([alert1, alert2])

    assert len(incidents) == 2
    assert incidents[0].incident_id != incidents[1].incident_id


def test_unrelated_components_at_same_time_not_correlated():
    """Verify independent components with unrelated errors are not correlated simply due to time proximity."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)

    alert1 = _create_alert("ALT-001", "billing-service", "Syntax error in batch parser", t0)
    alert2 = _create_alert("ALT-002", "analytics-etl", "Kafka offset commit delay", t0)

    correlator = IncidentCorrelator(window_seconds=300)
    incidents = correlator.correlate_batch([alert1, alert2])

    assert len(incidents) == 2
    assert incidents[0].components_affected == ["billing-service"]
    assert incidents[1].components_affected == ["analytics-etl"]


def test_service_dependencies_correlation():
    """Verify alerts across dependent services (postgres -> api) within window are correlated."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(seconds=30)

    alert_db = _create_alert(
        "ALT-DB",
        "postgres-db",
        "Container exited with code 137",
        t0,
        AlertSource.DOCKER,
        Severity.HIGH,
    )
    alert_api = _create_alert(
        "ALT-API",
        "api-service",
        "psycopg2.OperationalError: Connection refused",
        t1,
        AlertSource.LOGS,
        Severity.HIGH,
    )

    correlator = IncidentCorrelator(window_seconds=300)
    incidents = correlator.correlate_batch([alert_db, alert_api])

    assert len(incidents) == 1
    inc = incidents[0]
    assert "postgres-db" in inc.components_affected
    assert "api-service" in inc.components_affected


def test_cascading_failure_detection():
    """Verify cascading failure across 3 dependent tiers is detected and tracked in metadata."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(seconds=15)
    t2 = t0 + datetime.timedelta(seconds=30)

    alert_db = _create_alert(
        "ALT-1", "postgres", "Database process crashed", t0, AlertSource.DOCKER, Severity.HIGH
    )
    alert_api = _create_alert(
        "ALT-2",
        "backend-api",
        "Connection refused to database",
        t1,
        AlertSource.LOGS,
        Severity.HIGH,
    )
    alert_web = _create_alert(
        "ALT-3",
        "frontend-web",
        "Readiness probe 503 unavailable",
        t2,
        AlertSource.HEALTH_PROBE,
        Severity.HIGH,
    )

    correlator = IncidentCorrelator(window_seconds=300)
    incidents = correlator.correlate_batch([alert_db, alert_api, alert_web])

    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.metadata.get("cascading_failure") is True
    chain = inc.metadata.get("cascade_chain")
    assert chain is not None
    assert "postgres" in chain
    assert "backend-api" in chain


def test_duplicate_event_handling_increments_recurrence():
    """Verify duplicate alert with identical fingerprint increments recurrence without duplicating evidence."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(seconds=20)

    alert1 = _create_alert("ALT-DUP-1", "worker", "Queue backlog spike", t0)
    alert2 = _create_alert("ALT-DUP-1", "worker", "Queue backlog spike", t1)  # same fingerprint

    correlator = IncidentCorrelator(window_seconds=300)
    incidents = correlator.correlate_batch([alert1, alert2])

    assert len(incidents) == 1
    inc = incidents[0]
    assert len(inc.alerts) == 1
    assert inc.alerts[0].recurrence_count == 2
    # Evidence catalog should have 1 entry for this alert
    assert len(inc.evidence.evidence_catalog) == 1


def test_out_of_order_events():
    """Verify out-of-order alerts are properly ordered chronologically and correlated."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(seconds=50)

    alert1 = _create_alert("ALT-1", "app", "Initial spike", t0)
    alert2 = _create_alert("ALT-2", "app", "Secondary warning", t1)

    correlator = IncidentCorrelator(window_seconds=300)
    # Feed in reverse order
    incidents = correlator.correlate_batch([alert2, alert1])

    assert len(incidents) == 1
    inc = incidents[0]
    assert len(inc.alerts) == 2
    assert inc.created_at == t0


def test_empty_event_collection():
    """Verify empty alert list returns empty incident list."""
    correlator = IncidentCorrelator()
    assert correlator.correlate_batch([]) == []


def test_configurable_window_sizes():
    """Verify correlation window behavior adapts to custom window sizes."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(seconds=90)

    alert1 = _create_alert("ALT-1", "cache", "Eviction spike", t0)
    alert2 = _create_alert("ALT-2", "cache", "High latency", t1)

    # 60s window: should not correlate
    c60 = IncidentCorrelator(window_seconds=60)
    assert len(c60.correlate_batch([alert1, alert2])) == 2

    # 120s window: should correlate
    c120 = IncidentCorrelator(window_seconds=120)
    assert len(c120.correlate_batch([alert1, alert2])) == 1


def test_safe_payload_truncation():
    """Verify excessively large alert text is safely truncated in evidence catalog."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    huge_msg = "A" * 5000
    alert = _create_alert("ALT-HUGE", "logger", huge_msg, t0)

    correlator = IncidentCorrelator()
    incidents = correlator.correlate_batch([alert])

    assert len(incidents) == 1
    catalog_vals = list(incidents[0].evidence.evidence_catalog.values())
    assert len(catalog_vals) == 1
    assert len(catalog_vals[0]) <= 2500
    assert "[TRUNCATED]" in catalog_vals[0]
