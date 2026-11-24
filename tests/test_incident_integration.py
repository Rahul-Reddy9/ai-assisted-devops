# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""End-to-end integration tests for incident detection, correlation, two-tier RCA, remediation, and reporting."""

import datetime
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from devops_ai.cli import app
from devops_ai.incident.analyzer import IncidentAnalyzer
from devops_ai.incident.correlator import IncidentCorrelator
from devops_ai.incident.detector import IncidentDetector
from devops_ai.incident.models import (
    Alert,
    AlertSource,
    Category,
    Incident,
    IncidentStatus,
    RiskLevel,
    Severity,
)
from devops_ai.incident.reporter import IncidentReporter
from devops_ai.incident.repository import IncidentRepository

runner = CliRunner()


@pytest.fixture(autouse=True)
def setup_isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure all repositories use the test-isolated temporary directory."""
    orig_init = IncidentRepository.__init__

    def patched_init(self, storage_dir=None, *args, **kwargs):
        orig_init(self, storage_dir=storage_dir or tmp_path, *args, **kwargs)

    monkeypatch.setattr(IncidentRepository, "__init__", patched_init)


def test_end_to_end_incident_pipeline(tmp_path: Path):
    """Verify complete operational incident pipeline from detection to post-mortem export."""
    now = datetime.datetime.now(datetime.timezone.utc)
    repo = IncidentRepository(storage_dir=tmp_path)
    detector = IncidentDetector(repository=repo)

    # 1. Ingest Observability Alerts
    raw_alerts = [
        Alert(
            alert_id="ALT-E2E-001",
            source=AlertSource.LOGS,
            component="order-service",
            message="java.lang.OutOfMemoryError: Java heap space during batch processing",
            severity=Severity.HIGH,
            timestamp=now - datetime.timedelta(minutes=5),
            fingerprint="fp-order-oom-1",
        ),
        Alert(
            alert_id="ALT-E2E-002",
            source=AlertSource.DOCKER,
            component="order-service",
            message="Container exited with status 137 (OOMKilled: true)",
            severity=Severity.CRITICAL,
            timestamp=now - datetime.timedelta(minutes=4, seconds=45),
            fingerprint="fp-order-docker-1",
        ),
        Alert(
            alert_id="ALT-E2E-003",
            source=AlertSource.HEALTH_PROBE,
            component="order-service",
            message="Readiness probe failed HTTP 503: Service Unavailable",
            severity=Severity.MEDIUM,
            timestamp=now - datetime.timedelta(minutes=4, seconds=30),
            fingerprint="fp-order-probe-1",
        ),
    ]

    for a in raw_alerts:
        detector.ingest_alert(a)

    # 2. Correlate Alerts into an Incident
    correlator = IncidentCorrelator(window_seconds=600)
    incidents: list[Incident] = []
    for a in raw_alerts:
        inc = correlator.correlate_alert(a, incidents)
        if inc not in incidents:
            incidents.append(inc)

    assert len(incidents) == 1
    incident = incidents[0]

    assert incident is not None
    assert incident.components_affected == ["order-service"]
    assert len(incident.alerts) == 3
    assert len(incident.timeline) >= 3
    assert incident.severity == Severity.CRITICAL
    assert incident.status in (IncidentStatus.DETECTED, IncidentStatus.CORRELATED)

    # Enrich evidence catalog with confirmed container state
    incident.evidence.evidence_catalog["EVID-CONTAINER-01"] = (
        "Docker State.OOMKilled: true, container exited with status 137"
    )
    incident.evidence.evidence_catalog["EVID-LOG-01"] = (
        "java.lang.OutOfMemoryError: Java heap space"
    )

    # 3. Persist Initial Incident
    saved = repo.save(incident)
    assert saved.version == 1

    # 4. Perform Two-Tier Root Cause Analysis
    analyzer = IncidentAnalyzer(auto_tier2=False)
    analyzed = analyzer.analyze(saved, skip_tier2=True)
    assert analyzed.status == IncidentStatus.ANALYZED
    assert analyzed.root_cause is not None
    assert analyzed.root_cause.confidence > 0.50
    assert len(analyzed.recommendations) > 0

    # Ensure all recommendations are strictly advisory
    for rec in analyzed.recommendations:
        assert rec.risk_level in (RiskLevel.SAFE, RiskLevel.MODERATE, RiskLevel.DANGEROUS)
        assert "ADVISORY ONLY" in rec.advisory_warning
        assert rec.verification_step is not None

    repo.save(analyzed)

    # 5. Incident Resolution and Closure Lifecycle
    analyzed.status = IncidentStatus.RESOLVED
    analyzed.resolved_at = datetime.datetime.now(datetime.timezone.utc)
    analyzed.resolution_notes = (
        "Scaled pod memory limit from 512Mi to 1Gi. Monitored garbage collection for 15 minutes."
    )
    resolved = repo.save(analyzed)
    assert resolved.status == IncidentStatus.RESOLVED

    resolved.status = IncidentStatus.CLOSED
    resolved.closed_at = datetime.datetime.now(datetime.timezone.utc)
    closed = repo.save(resolved)
    assert closed.status == IncidentStatus.CLOSED

    # 6. Export Post-Mortem Reports
    md_path = tmp_path / "post_mortem.md"
    md_content = IncidentReporter.export_report(closed, format="markdown", output_path=md_path)
    assert md_path.exists()
    assert f"# Incident Post-Mortem: {closed.incident_id}" in md_content
    assert "## Primary Root Cause Analysis" in md_content
    assert "## Remediation Advisory" in md_content
    assert "> [!WARNING]" in md_content
    assert "ADVISORY ONLY" in md_content
    assert "Scaled pod memory limit" in md_content

    json_path = tmp_path / "incident.json"
    json_content = IncidentReporter.export_report(closed, format="json", output_path=json_path)
    assert json_path.exists()
    loaded = json.loads(json_content)
    assert loaded["incident_id"] == closed.incident_id
    assert loaded["status"] == "CLOSED"
    assert loaded["resolution_notes"] is not None


def test_end_to_end_cli_lifecycle_workflow(tmp_path: Path):
    """Verify complete incident lifecycle executed entirely through DevOps AI CLI commands."""
    now = datetime.datetime.now(datetime.timezone.utc)
    repo = IncidentRepository(storage_dir=tmp_path)

    # 1. Seed incident in repository
    inc = Incident(
        incident_id="INC-20261124-E2E-CLI",
        title="Payment Service Degradation and Connection Timeout",
        category=Category.APPLICATION,
        severity=Severity.HIGH,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
        components_affected=["payment-service", "redis"],
        alerts=[
            Alert(
                alert_id="ALT-REDIS-1",
                source=AlertSource.LOGS,
                component="payment-service",
                message="redis.exceptions.ConnectionError: Connection refused",
                severity=Severity.HIGH,
                timestamp=now,
                fingerprint="fp-redis-1",
            )
        ],
    )
    repo.save(inc)

    # 2. View incident via CLI
    show_res = runner.invoke(app, ["incident", "show", "INC-20261124-E2E-CLI"])
    assert show_res.exit_code == 0
    assert "INC-20261124-E2E-CLI" in show_res.output

    # 3. Analyze incident via CLI
    analyze_res = runner.invoke(
        app, ["incident", "analyze", "INC-20261124-E2E-CLI", "--skip-tier2"]
    )
    assert analyze_res.exit_code == 0
    assert "ANALYZED" in analyze_res.output

    # 4. Inspect RCA via CLI --json
    rca_res = runner.invoke(
        app, ["incident", "rca", "INC-20261124-E2E-CLI", "--skip-tier2", "--json"]
    )
    assert rca_res.exit_code == 0
    rca_data = json.loads(rca_res.output)
    assert rca_data["incident_id"] == "INC-20261124-E2E-CLI"
    assert rca_data["status"] == "ANALYZED"
    assert len(rca_data["recommendations"]) > 0

    # 5. Resolve incident via CLI
    resolve_res = runner.invoke(
        app,
        [
            "incident",
            "resolve",
            "INC-20261124-E2E-CLI",
            "--notes",
            "Redis cluster failover recovered; application reconnected cleanly",
        ],
    )
    assert resolve_res.exit_code == 0
    assert "RESOLVED" in resolve_res.output

    # 6. Close incident via CLI
    close_res = runner.invoke(
        app,
        [
            "incident",
            "close",
            "INC-20261124-E2E-CLI",
            "--notes",
            "Incident review verified; failover alarm threshold lowered",
        ],
    )
    assert close_res.exit_code == 0
    assert "CLOSED" in close_res.output

    # 7. Generate and export post-mortem report via CLI
    out_md = tmp_path / "cli_postmortem.md"
    report_res = runner.invoke(
        app,
        [
            "incident",
            "report",
            "INC-20261124-E2E-CLI",
            "--format",
            "markdown",
            "--output",
            str(out_md),
        ],
    )
    assert report_res.exit_code == 0
    assert "Post-mortem report exported successfully" in report_res.output
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "# Incident Post-Mortem: INC-20261124-E2E-CLI" in content
    assert "Redis cluster failover recovered" in content
    assert "> [!WARNING]" in content
