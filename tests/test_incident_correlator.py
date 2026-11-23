# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying multi-source incident correlation and sliding window behavior."""

import datetime
import pytest
from devops_ai.incident.correlator import IncidentCorrelator, normalize_timestamp
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


# --- Audit Finding F-01 Tests: Timestamp Normalization ---


def test_f01_timestamp_normalization_naive_vs_aware():
    """Verify naive and aware timestamps are normalized to UTC and compared without TypeError."""
    naive_dt = datetime.datetime(2026, 11, 23, 12, 0, 0)  # naive
    aware_dt = datetime.datetime(2026, 11, 23, 12, 2, 0, tzinfo=datetime.timezone.utc)  # aware

    norm_naive = normalize_timestamp(naive_dt)
    norm_aware = normalize_timestamp(aware_dt)

    assert norm_naive.tzinfo == datetime.timezone.utc
    assert norm_aware.tzinfo == datetime.timezone.utc
    delta = (norm_aware - norm_naive).total_seconds()
    assert delta == 120.0

    # Correlator handles naive and aware alerts seamlessly
    alert1 = _create_alert("ALT-NAIVE", "api", "Msg 1", naive_dt)
    alert2 = _create_alert("ALT-AWARE", "api", "Msg 2", aware_dt)

    correlator = IncidentCorrelator(window_seconds=300)
    incidents = correlator.correlate_batch([alert1, alert2])
    assert len(incidents) == 1
    assert len(incidents[0].alerts) == 2


def test_f01_timestamp_normalization_non_utc_offsets():
    """Verify timestamps with non-UTC timezone offsets representing the same instant match."""
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    t_ist = datetime.datetime(2026, 11, 23, 17, 30, 0, tzinfo=ist)
    t_utc = datetime.datetime(2026, 11, 23, 12, 0, 0, tzinfo=datetime.timezone.utc)

    norm_ist = normalize_timestamp(t_ist)
    norm_utc = normalize_timestamp(t_utc)

    assert norm_ist == norm_utc
    assert abs((norm_ist - norm_utc).total_seconds()) == 0.0


def test_f01_timestamp_normalization_iso_strings_and_invalid_inputs():
    """Verify ISO strings are parsed to UTC and invalid timestamps raise ValueError."""
    iso_utc = "2026-11-23T12:00:00Z"
    iso_offset = "2026-11-23T17:30:00+05:30"

    norm1 = normalize_timestamp(iso_utc)
    norm2 = normalize_timestamp(iso_offset)

    assert norm1.tzinfo == datetime.timezone.utc
    assert norm1 == norm2

    # Invalid timestamp strings or None must raise ValueError
    with pytest.raises(ValueError, match="Timestamp cannot be None"):
        normalize_timestamp(None)

    with pytest.raises(ValueError, match="Invalid timestamp string"):
        normalize_timestamp("invalid-date-format")

    with pytest.raises(ValueError, match="Expected datetime object"):
        normalize_timestamp(123456789)


def test_f01_sliding_window_boundary_conditions():
    """Verify events exactly at boundary (diff == window) correlate, and just outside (diff > window) do not."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t_boundary = t0 + datetime.timedelta(seconds=300)
    t_outside = t0 + datetime.timedelta(seconds=301)

    c = IncidentCorrelator(window_seconds=300)

    # Exactly at boundary: total_seconds == 300 <= 300
    a1 = _create_alert("A1", "web", "msg1", t0)
    a2 = _create_alert("A2", "web", "msg2", t_boundary)
    assert c.are_alerts_correlatable(a1, a2) is True

    # Just outside boundary: total_seconds == 301 > 300
    a3 = _create_alert("A3", "web", "msg3", t_outside)
    assert c.are_alerts_correlatable(a1, a3) is False


# --- Audit Finding F-02 Tests: Component Matching Boundaries ---


def test_f02_delimiter_aware_component_matching():
    """Verify delimiter-aware component instance matching and false-positive prevention."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    c = IncidentCorrelator(window_seconds=300)

    # Valid matches: web-app & web-app-1, api & api-v2, postgres & postgres-primary
    assert (
        c.are_alerts_correlatable(
            _create_alert("A1", "web-app", "err", t0),
            _create_alert("A2", "web-app-1", "err", t0),
        )
        is True
    )

    assert (
        c.are_alerts_correlatable(
            _create_alert("A3", "api", "err", t0),
            _create_alert("A4", "api-v2", "err", t0),
        )
        is True
    )

    assert (
        c.are_alerts_correlatable(
            _create_alert("A5", "postgres", "err", t0),
            _create_alert("A6", "postgres-primary", "err", t0),
        )
        is True
    )

    # False-positive substring containment matches MUST be rejected:
    # 'api' vs 'capital-mgmt'
    assert (
        c.are_alerts_correlatable(
            _create_alert("A7", "api", "err", t0),
            _create_alert("A8", "capital-mgmt", "err", t0),
        )
        is False
    )

    # 'app' vs 'wrapper'
    assert (
        c.are_alerts_correlatable(
            _create_alert("A9", "app", "err", t0),
            _create_alert("A10", "wrapper", "err", t0),
        )
        is False
    )

    # 'auth' vs 'author-service'
    assert (
        c.are_alerts_correlatable(
            _create_alert("A11", "auth", "err", t0),
            _create_alert("A12", "author-service", "err", t0),
        )
        is False
    )


# --- Audit Finding F-03 Tests: Generic Keyword Correlation Prevention ---


def test_f03_generic_keywords_alone_do_not_correlate_independent_services():
    """Verify generic error phrases (econnrefused, rate limit, oom) alone do not correlate independent services."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    c = IncidentCorrelator(window_seconds=300)

    # Two independent services in different regions/components with 'connection refused'
    a_us = _create_alert(
        "A-US",
        "billing-us-east",
        "Connection refused to port 8080",
        t0,
        raw_data={"region": "us-east-1"},
    )
    a_eu = _create_alert(
        "A-EU",
        "shipping-eu-west",
        "Connection refused to port 9090",
        t0,
        raw_data={"region": "eu-west-1"},
    )

    # Must NOT correlate because they share no identity, pod, container, or dependency
    assert c.are_alerts_correlatable(a_us, a_eu) is False
    incidents = c.correlate_batch([a_us, a_eu])
    assert len(incidents) == 2

    # Two services with unrelated HTTP 429
    a_429_1 = _create_alert("A-429-1", "user-service", "HTTP 429 Too Many Requests", t0)
    a_429_2 = _create_alert("A-429-2", "catalog-service", "HTTP 429 Too Many Requests", t0)
    assert c.are_alerts_correlatable(a_429_1, a_429_2) is False

    # Two services sharing a container ID or pod name DO correlate
    a_shared_c1 = _create_alert(
        "A-C1", "worker", "Crash", t0, raw_data={"container_id": "c-998877"}
    )
    a_shared_c2 = _create_alert(
        "A-C2", "logger", "Crash", t0, raw_data={"container_id": "c-998877"}
    )
    assert c.are_alerts_correlatable(a_shared_c1, a_shared_c2) is True

    # Two services sharing a verified database dependency DO correlate
    a_db = _create_alert("A-DB", "postgres", "Process died", t0)
    a_api = _create_alert("A-API", "api", "psycopg2 connection error", t0)
    assert c.are_alerts_correlatable(a_db, a_api) is True


# --- Audit Finding F-08 Tests: Cascading Failure Chain Precision ---


def test_f08_cascade_chain_excludes_unrelated_early_alerts():
    """Verify cascade chain only includes participating dependency subgraph and excludes unrelated earlier components."""
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = t0 + datetime.timedelta(minutes=10)  # 10:10
    t2 = t0 + datetime.timedelta(minutes=20)  # 10:20

    # Scenario:
    # analytics reports unrelated alert at 10:00
    # postgres fails at 10:10
    # api reports connection failure at 10:20
    alert_analytics = _create_alert("ALT-0", "analytics", "Slow query execution warning", t0)
    alert_postgres = _create_alert(
        "ALT-1", "postgres", "Database crashed with SIGSEGV", t1, AlertSource.DOCKER, Severity.HIGH
    )
    alert_api = _create_alert(
        "ALT-2",
        "api",
        "psycopg2.OperationalError: could not connect to server",
        t2,
        AlertSource.LOGS,
        Severity.HIGH,
    )

    c = IncidentCorrelator(window_seconds=1800)  # 30-minute correlation window
    # All 3 alerts processed
    chain = c._detect_cascade(
        components=["analytics", "postgres", "api"],
        alerts=[alert_analytics, alert_postgres, alert_api],
    )

    # Expected cascade chain: postgres -> api. Analytics must NOT be in the chain!
    assert chain == ["postgres", "api"]
    assert "analytics" not in chain


def test_f08_multi_hop_cascade_chain_causality():
    """Verify 3-hop cascade chain (postgres -> backend -> frontend) preserves causal ordering with unrelated alerts."""
    t_early = datetime.datetime(2026, 11, 23, 9, 55, 0, tzinfo=datetime.timezone.utc)
    t0 = datetime.datetime(2026, 11, 23, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2026, 11, 23, 10, 5, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2026, 11, 23, 10, 10, 0, tzinfo=datetime.timezone.utc)

    alert_unrelated = _create_alert("ALT-U", "reporting-worker", "Memory warning", t_early)
    alert_db = _create_alert("ALT-DB", "postgres", "Database down", t0)
    alert_be = _create_alert("ALT-BE", "backend", "Database connection lost", t1)
    alert_fe = _create_alert("ALT-FE", "frontend", "502 Bad Gateway from backend", t2)

    c = IncidentCorrelator(window_seconds=1800)
    chain = c._detect_cascade(
        components=["reporting-worker", "postgres", "backend", "frontend"],
        alerts=[alert_unrelated, alert_db, alert_be, alert_fe],
    )

    assert chain == ["postgres", "backend", "frontend"]
    assert "reporting-worker" not in chain
