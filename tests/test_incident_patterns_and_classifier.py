# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying failure patterns, OOM/SIGKILL distinction, and deterministic classification."""

import datetime
from devops_ai.incident.classifier import IncidentClassifier
from devops_ai.incident.models import (
    Alert,
    AlertSource,
    Category,
    CorrelatedEvidence,
    Incident,
    IncidentStatus,
    Severity,
)
from devops_ai.incident.patterns import FailurePatternCatalog


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


# --- 1. OOM vs SIGKILL Tests ---


def test_pattern_ambiguous_exit_code_137_without_oom_corroboration():
    """Verify exit code 137 alone is diagnosed as CONTAINER_SIGKILL_TERMINATION (not confirmed OOM)."""
    alert = _make_alert(
        "ALT-137", AlertSource.DOCKER, "web", "Container Exited (137) 2 minutes ago", Severity.HIGH
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-DOCKER-01": "Container Exited (137)"},
        container_states=[{"status": "Exited (137)", "exit_code": 137, "oom_killed": False}],
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert len(matches) >= 1
    top = matches[0]
    assert top.pattern_id == "CONTAINER_SIGKILL_TERMINATION"
    assert top.is_confirmed is False
    assert top.confidence_breakdown.final_score == 0.51
    assert (
        "Missing Kubernetes pod termination reason OOMKilled" in top.description
        or len(top.missing_evidence) > 0
    )


def test_pattern_confirmed_oom_with_docker_oomkilled_flag():
    """Verify exit code 137 with Docker State.OOMKilled: true is diagnosed as confirmed CONTAINER_OOM_KILLED."""
    alert = _make_alert(
        "ALT-OOM", AlertSource.DOCKER, "worker", "Container Exited (137)", Severity.HIGH
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-DOCKER-01": "Container Exited (137)"},
        container_states=[{"status": "Exited (137)", "exit_code": 137, "oom_killed": True}],
        metric_anomalies={"worker": 94.5},
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert len(matches) >= 1
    top = matches[0]
    assert top.pattern_id == "CONTAINER_OOM_KILLED"
    assert top.is_confirmed is True
    assert top.confidence_breakdown.final_score >= 0.95


def test_pattern_confirmed_oom_with_k8s_termination_reason():
    """Verify Kubernetes OOMKilled termination reason confirms CONTAINER_OOM_KILLED."""
    alert = _make_alert(
        "ALT-K8S-OOM", AlertSource.KUBERNETES, "payment-pod", "Pod exited code 137", Severity.HIGH
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-K8S-01": "Pod terminated with exit code 137"},
        k8s_events=[{"reason": "OOMKilled", "message": "Container payment exceeded memory limit"}],
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert len(matches) >= 1
    top = matches[0]
    assert top.pattern_id == "CONTAINER_OOM_KILLED"
    assert top.is_confirmed is True


def test_pattern_confirmed_oom_with_kernel_dmesg_log():
    """Verify kernel Out of memory regex in log confirms CONTAINER_OOM_KILLED."""
    alert = _make_alert(
        "ALT-DMESG",
        AlertSource.LOGS,
        "host-node",
        "kernel: Out of memory: Kill process 1234 (python)",
        Severity.HIGH,
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={
            "EVID-LOG-01": "kernel: Out of memory: Kill process 1234 (python) score 950",
            "EVID-DOCKER-01": "Container exited (137)",
        },
        container_states=[{"status": "Exited (137)", "exit_code": 137}],
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert len(matches) >= 1
    top = matches[0]
    assert top.pattern_id == "CONTAINER_OOM_KILLED"
    assert top.is_confirmed is True


def test_pattern_high_memory_utilization_alone_does_not_confirm_oom():
    """Verify high memory utilization (>90%) without exit 137 or OOMKilled is diagnosed as MEMORY_EXHAUSTION."""
    alert = _make_alert(
        "ALT-MEM",
        AlertSource.METRICS,
        "cache-svc",
        "Memory utilization high: 95.2%",
        Severity.MEDIUM,
        raw={"memory_percent": 95.2},
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-METRIC-01": "memory_percent: 95.2%"},
        metric_anomalies={"cache-svc": 95.2},
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert len(matches) >= 1
    top = matches[0]
    assert top.pattern_id == "MEMORY_EXHAUSTION"
    assert "does NOT confirm an OOM kill" in top.description


# --- 2. Other Failure Patterns Tests ---


def test_pattern_k8s_crash_loop_backoff():
    """Verify CrashLoopBackOff detection with log corroboration."""
    alert = _make_alert(
        "ALT-CLB", AlertSource.KUBERNETES, "api-pod", "Pod entering CrashLoopBackOff", Severity.HIGH
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-K8S-01": "Back-off 40s restarting failed container=api"},
        k8s_events=[{"reason": "CrashLoopBackOff", "restarts": 4}],
        log_snippets=["Fatal error during bootstrap: Traceback (most recent call last)"],
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert any(m.pattern_id == "K8S_CRASH_LOOP_BACKOFF" and m.is_confirmed for m in matches)


def test_pattern_k8s_image_pull_backoff():
    """Verify ImagePullBackOff detection."""
    alert = _make_alert(
        "ALT-IMG",
        AlertSource.KUBERNETES,
        "worker-pod",
        "ErrImagePull: image not found",
        Severity.HIGH,
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={
            "EVID-K8S-01": "Failed to pull image registry.example.com/app:v9.9: ImagePullBackOff"
        },
        k8s_events=[{"reason": "ImagePullBackOff", "message": "Failed to pull image"}],
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert any(m.pattern_id == "K8S_IMAGE_PULL_BACKOFF" and m.is_confirmed for m in matches)


def test_pattern_db_connection_refused_positive_and_contradiction():
    """Verify DB ConnectionRefused matches and applies contradiction penalty when probe is OK."""
    alert = _make_alert(
        "ALT-DB",
        AlertSource.LOGS,
        "backend",
        "psycopg2.OperationalError: connection refused to port 5432",
        Severity.HIGH,
    )

    # Positive match with probe failure
    evid_corrob = CorrelatedEvidence(
        evidence_catalog={"EVID-LOG-01": "psycopg2.OperationalError: connection refused"},
        probe_failures=[{"status_code": 503}],
    )
    m_pos = FailurePatternCatalog.match_patterns([alert], evid_corrob)
    db_pos = [m for m in m_pos if m.pattern_id == "DB_CONNECTION_REFUSED"]
    assert len(db_pos) == 1
    assert db_pos[0].confidence_breakdown.final_score == 0.83  # 0.80 * 0.85 + 0.15 = 0.83

    # Contradictory probe reports HTTP 200 OK
    alert_probe_ok = _make_alert(
        "ALT-PROBE-OK",
        AlertSource.HEALTH_PROBE,
        "backend",
        "Probe OK",
        Severity.INFO,
        raw={"status_code": 200},
    )
    m_contra = FailurePatternCatalog.match_patterns([alert, alert_probe_ok], evid_corrob)
    db_contra = [m for m in m_contra if m.pattern_id == "DB_CONNECTION_REFUSED"]
    assert len(db_contra) == 1
    assert db_contra[0].confidence_breakdown.contradiction_penalty == 0.40
    assert db_contra[0].confidence_breakdown.final_score < 0.50


def test_pattern_cpu_throttling():
    """Verify CPU throttling detection from metrics and cgroup logs."""
    alert = _make_alert(
        "ALT-CPU",
        AlertSource.METRICS,
        "etl-worker",
        "CPU spike: 96.0%",
        Severity.MEDIUM,
        raw={"cpu_percent": 96.0},
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-METRIC-01": "cpu_percent: 96.0"},
        metric_anomalies={"etl-worker": 96.0},
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert any(m.pattern_id == "CPU_THROTTLING" for m in matches)


def test_pattern_readiness_probe_failure():
    """Verify HTTP 503 readiness probe failure detection."""
    alert = _make_alert(
        "ALT-PROBE",
        AlertSource.HEALTH_PROBE,
        "web-service",
        "Readiness probe failed with HTTP 503",
        Severity.HIGH,
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={"EVID-PROBE-01": "Readiness probe failed: HTTP 503 Service Unavailable"},
        probe_failures=[{"status_code": 503, "checks": {"db": {"status": "error"}}}],
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert any(m.pattern_id == "READINESS_PROBE_FAILURE" for m in matches)


def test_pattern_upstream_request_timeout():
    """Verify upstream request timeout pattern."""
    alert = _make_alert(
        "ALT-TIMEOUT",
        AlertSource.LOGS,
        "gateway",
        "Upstream request timeout after 30.0s",
        Severity.MEDIUM,
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={
            "EVID-LOG-01": "504 Gateway Time-out: upstream request timeout after 30000ms"
        },
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert any(m.pattern_id == "UPSTREAM_REQUEST_TIMEOUT" for m in matches)


def test_pattern_llm_rate_limiting():
    """Verify LLM HTTP 429 rate limit pattern."""
    alert = _make_alert(
        "ALT-LLM-429",
        AlertSource.LOGS,
        "ai-agent",
        "Provider returned 429 Too Many Requests: Rate limit exceeded",
        Severity.MEDIUM,
    )
    evidence = CorrelatedEvidence(
        evidence_catalog={
            "EVID-LOG-01": "Provider returned 429 Too Many Requests: TPM quota exceeded"
        },
    )

    matches = FailurePatternCatalog.match_patterns([alert], evidence)
    assert any(m.pattern_id == "LLM_RATE_LIMITING" for m in matches)


# --- 3. Classifier Tests ---


def test_classifier_all_categories_supported():
    """Verify classifier correctly maps each category based on evidence and patterns."""
    now = datetime.datetime.now(datetime.timezone.utc)
    base_inc = Incident(
        incident_id="INC-CAT",
        version=1,
        title="Category Test",
        category=Category.UNKNOWN,
        severity=Severity.MEDIUM,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
    )

    # Provider
    base_inc.alerts = [_make_alert("A1", AlertSource.LOGS, "llm-gemini", "429 Too Many Requests")]
    c_prov = IncidentClassifier.classify(base_inc)
    assert c_prov.category == Category.PROVIDER

    # Container
    base_inc.alerts = [_make_alert("A2", AlertSource.DOCKER, "web", "Container died")]
    c_cont = IncidentClassifier.classify(base_inc)
    assert c_cont.category == Category.CONTAINER

    # Infrastructure
    base_inc.alerts = [_make_alert("A3", AlertSource.METRICS, "node-1", "CPU load 99%")]
    c_infra = IncidentClassifier.classify(base_inc)
    assert c_infra.category == Category.INFRASTRUCTURE

    # Application
    base_inc.alerts = [_make_alert("A4", AlertSource.HEALTH_PROBE, "api", "503 probe failure")]
    c_app = IncidentClassifier.classify(base_inc)
    assert c_app.category == Category.APPLICATION


def test_classifier_prevention_of_unjustified_critical_severity():
    """Verify single alert claiming CRITICAL without strong evidence is clamped to HIGH."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert = _make_alert(
        "A-CRIT", AlertSource.LOGS, "single-app", "Unverified critical assertion", Severity.CRITICAL
    )
    inc = Incident(
        incident_id="INC-CLAMP",
        version=1,
        title="Clamp Test",
        category=Category.UNKNOWN,
        severity=Severity.CRITICAL,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
        alerts=[alert],
        components_affected=["single-app"],
    )

    res = IncidentClassifier.classify(inc)
    assert res.severity == Severity.HIGH
    assert res.is_critical_justified is False
    assert "clamped to HIGH" in res.reason


def test_classifier_justified_critical_severity_on_multi_component_outage():
    """Verify multi-component outage with high-confidence pattern justifies CRITICAL."""
    now = datetime.datetime.now(datetime.timezone.utc)
    alert1 = _make_alert("A1", AlertSource.DOCKER, "db", "OOM killed", Severity.HIGH)
    alert2 = _make_alert("A2", AlertSource.LOGS, "backend", "connection refused", Severity.HIGH)
    inc = Incident(
        incident_id="INC-OUTAGE",
        version=1,
        title="Outage Test",
        category=Category.APPLICATION,
        severity=Severity.HIGH,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
        alerts=[alert1, alert2],
        components_affected=["db", "backend"],
        metadata={"cascading_failure": True},
    )

    # High-confidence pattern match
    pm = FailurePatternCatalog.match_patterns(
        [alert1],
        CorrelatedEvidence(
            evidence_catalog={"EVID-1": "OOM"},
            container_states=[{"status": "Exited (137)", "oom_killed": True}],
        ),
    )[0]

    res = IncidentClassifier.classify(inc, primary_pattern=pm)
    assert res.severity == Severity.CRITICAL
    assert res.is_critical_justified is True
    assert res.blast_radius == "systemic"
