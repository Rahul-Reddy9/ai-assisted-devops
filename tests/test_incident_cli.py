# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit and functional tests for DevOps AI incident CLI subcommands."""

import datetime
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from devops_ai.cli import app
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
)
from devops_ai.incident.repository import IncidentRepository

runner = CliRunner()


@pytest.fixture(autouse=True)
def setup_isolated_incident_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure all CLI invocations use an isolated temporary storage directory."""
    orig_init = IncidentRepository.__init__

    def patched_init(self, storage_dir=None, *args, **kwargs):
        orig_init(self, storage_dir=storage_dir or tmp_path, *args, **kwargs)

    monkeypatch.setattr(IncidentRepository, "__init__", patched_init)


@pytest.fixture
def seeded_repo(tmp_path: Path) -> IncidentRepository:
    """Create and seed an isolated repository with sample test incidents."""
    repo = IncidentRepository(storage_dir=tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Active OOM Incident
    inc1 = Incident(
        incident_id="INC-20261124-0001",
        title="OOMKilled event on payment-gateway",
        category=Category.CONTAINER,
        severity=Severity.CRITICAL,
        status=IncidentStatus.DETECTED,
        created_at=now - datetime.timedelta(minutes=30),
        updated_at=now - datetime.timedelta(minutes=30),
        components_affected=["payment-gateway"],
        alerts=[
            Alert(
                alert_id="ALT-1",
                source=AlertSource.DOCKER,
                component="payment-gateway",
                message="Container exited with status 137 (OOMKilled: true)",
                severity=Severity.CRITICAL,
                timestamp=now - datetime.timedelta(minutes=30),
                fingerprint="fp1",
            )
        ],
        evidence=CorrelatedEvidence(
            evidence_catalog={
                "EVID-CONTAINER-01": "Container exit code 137 (OOMKilled: True)",
            }
        ),
    )
    repo.save(inc1)

    # 2. Resolved Application Incident
    inc2 = Incident(
        incident_id="INC-20261124-0002",
        title="Database connection timeout on auth-service",
        category=Category.APPLICATION,
        severity=Severity.HIGH,
        status=IncidentStatus.RESOLVED,
        created_at=now - datetime.timedelta(hours=2),
        updated_at=now - datetime.timedelta(hours=1),
        resolved_at=now - datetime.timedelta(hours=1),
        resolution_notes="Postgres connection pool exhausted; pool size increased to 50",
        components_affected=["auth-service", "postgres"],
        root_cause=RootCauseHypothesis(
            title="Database Connection Pool Exhaustion",
            description="Exceeded max connections on primary database.",
            confidence=0.88,
            tier="tier_1_deterministic",
            evidence_refs=[],
            is_confirmed=True,
        ),
        recommendations=[
            RemediationAction(
                action_id="ACT-P1",
                title="Scale Database Connection Limit",
                description="Increase max_connections parameter.",
                action_type="permanent_fix",
                risk_level=RiskLevel.MODERATE,
                command="kubectl set env deployment/auth-service DB_POOL_SIZE=50",
                verification_step="Check active connections metric.",
            )
        ],
    )
    repo.save(inc2)

    return repo


def test_incident_cli_help():
    """Verify `devops-ai incident --help` displays all subcommands."""
    result = runner.invoke(app, ["incident", "--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
    assert "list" in result.output
    assert "show" in result.output
    assert "resolve" in result.output
    assert "close" in result.output
    assert "report" in result.output
    assert "rca" in result.output


def test_incident_list_empty():
    """Verify `devops-ai incident list` on empty repository returns clean message."""
    result = runner.invoke(app, ["incident", "list"])
    assert result.exit_code == 0
    assert "No incidents matching the specified criteria were found" in result.output


def test_incident_list_json_empty():
    """Verify `devops-ai incident list --json` returns valid empty array."""
    result = runner.invoke(app, ["incident", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == []


def test_incident_list_with_incidents(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident list` displays formatted table of incidents."""
    result = runner.invoke(app, ["incident", "list"], env={"COLUMNS": "140"})
    assert result.exit_code == 0
    assert "INC-20261124-0001" in result.output
    assert "INC-20261124-0002" in result.output
    assert "payment-gateway" in result.output
    assert "CRITICAL" in result.output
    assert "RESOLVED" in result.output


def test_incident_list_filtering(seeded_repo: IncidentRepository):
    """Verify filtering by status, severity, category, and active_only."""
    # Filter by status DETECTED
    res1 = runner.invoke(app, ["incident", "list", "--status", "DETECTED"], env={"COLUMNS": "140"})
    assert res1.exit_code == 0
    assert "INC-20261124-0001" in res1.output
    assert "INC-20261124-0002" not in res1.output

    # Filter active only
    res2 = runner.invoke(app, ["incident", "list", "--active-only"], env={"COLUMNS": "140"})
    assert res2.exit_code == 0
    assert "INC-20261124-0001" in res2.output
    assert "INC-20261124-0002" not in res2.output

    # Filter by category
    res3 = runner.invoke(
        app, ["incident", "list", "--category", "APPLICATION"], env={"COLUMNS": "140"}
    )
    assert res3.exit_code == 0
    assert "INC-20261124-0002" in res3.output
    assert "INC-20261124-0001" not in res3.output


def test_incident_list_invalid_status_and_severity():
    """Verify invalid status or severity flags return exit code 2."""
    res1 = runner.invoke(app, ["incident", "list", "--status", "INVALID_STATUS"])
    assert res1.exit_code == 2
    assert "Invalid status: 'INVALID_STATUS'" in res1.output

    res2 = runner.invoke(app, ["incident", "list", "--severity", "SUPER_HIGH"])
    assert res2.exit_code == 2
    assert "Invalid severity: 'SUPER_HIGH'" in res2.output


def test_incident_show_existing(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident show` displays detailed incident overview."""
    result = runner.invoke(app, ["incident", "show", "INC-20261124-0002"])
    assert result.exit_code == 0
    assert "INC-20261124-0002" in result.output
    assert "Database connection timeout" in result.output
    assert "Primary Root Cause Analysis" in result.output
    assert "Database Connection Pool Exhaustion" in result.output
    assert "ACT-P1" in result.output


def test_incident_show_json(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident show --json` emits valid JSON model."""
    result = runner.invoke(app, ["incident", "show", "INC-20261124-0002", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["incident_id"] == "INC-20261124-0002"
    assert data["severity"] == "HIGH"
    assert data["status"] == "RESOLVED"
    assert data["root_cause"]["title"] == "Database Connection Pool Exhaustion"


def test_incident_show_not_found():
    """Verify `devops-ai incident show` with non-existent ID exits with code 1."""
    result = runner.invoke(app, ["incident", "show", "INC-NONEXISTENT"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_incident_analyze_success(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident analyze` performs RCA, transitions state, and updates repo."""
    result = runner.invoke(app, ["incident", "analyze", "INC-20261124-0001", "--skip-tier2"])
    assert result.exit_code == 0
    assert "INC-20261124-0001" in result.output
    assert "ANALYZED" in result.output

    # Verify repository state was persisted
    updated = seeded_repo.get("INC-20261124-0001")
    assert updated.status == IncidentStatus.ANALYZED
    assert updated.root_cause is not None
    assert "SIGKILL" in updated.root_cause.title or "OOMKilled" in updated.root_cause.title


def test_incident_analyze_json(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident analyze --json` outputs analyzed model as JSON."""
    result = runner.invoke(
        app, ["incident", "analyze", "INC-20261124-0001", "--skip-tier2", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["incident_id"] == "INC-20261124-0001"
    assert data["status"] == "ANALYZED"
    assert data["root_cause"] is not None


def test_incident_analyze_not_found():
    """Verify `devops-ai incident analyze` with non-existent ID exits with code 1."""
    result = runner.invoke(app, ["incident", "analyze", "INC-MISSING"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_incident_rca_success(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident rca` executes targeted analysis and shows diagnosis."""
    result = runner.invoke(app, ["incident", "rca", "INC-20261124-0001", "--skip-tier2"])
    assert result.exit_code == 0
    assert "Root Cause Diagnosis" in result.output
    assert "OOMKilled" in result.output
    assert "Remediation Advisory" in result.output


def test_incident_rca_json(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident rca --json` emits targeted RCA payload."""
    result = runner.invoke(app, ["incident", "rca", "INC-20261124-0001", "--skip-tier2", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["incident_id"] == "INC-20261124-0001"
    assert "root_cause" in data
    assert "recommendations" in data
    assert data["root_cause"] is not None


def test_incident_resolve_command(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident resolve` marks incident resolved with notes."""
    result = runner.invoke(
        app,
        [
            "incident",
            "resolve",
            "INC-20261124-0001",
            "--notes",
            "Memory limit increased to 2Gi; pods restarted cleanly",
        ],
    )
    assert result.exit_code == 0
    assert "RESOLVED" in result.output

    updated = seeded_repo.get("INC-20261124-0001")
    assert updated.status == IncidentStatus.RESOLVED
    assert updated.resolved_at is not None
    assert "Memory limit increased" in updated.resolution_notes
    assert any("Incident resolved by operator" in t.event for t in updated.timeline)


def test_incident_resolve_json(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident resolve --json` emits updated JSON model."""
    result = runner.invoke(
        app,
        ["incident", "resolve", "INC-20261124-0001", "--notes", "Resolved via CLI", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "RESOLVED"
    assert data["resolution_notes"] == "Resolved via CLI"


def test_incident_close_command(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident close` marks incident closed with notes."""
    result = runner.invoke(
        app,
        [
            "incident",
            "close",
            "INC-20261124-0002",
            "--notes",
            "Post-incident review completed; Jira ticket closed",
        ],
    )
    assert result.exit_code == 0
    assert "CLOSED" in result.output

    updated = seeded_repo.get("INC-20261124-0002")
    assert updated.status == IncidentStatus.CLOSED
    assert updated.closed_at is not None
    assert "Post-incident review completed" in updated.resolution_notes
    assert any("Incident closed by operator" in t.event for t in updated.timeline)


def test_incident_close_json(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident close --json` emits updated JSON model."""
    result = runner.invoke(
        app,
        ["incident", "close", "INC-20261124-0002", "--notes", "Closed via CLI", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "CLOSED"


def test_incident_report_markdown_stdout(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident report` generates Markdown report to stdout."""
    result = runner.invoke(app, ["incident", "report", "INC-20261124-0002", "--format", "markdown"])
    assert result.exit_code == 0
    assert "# Incident Post-Mortem: INC-20261124-0002" in result.output
    assert "## Executive Summary" in result.output
    assert "Database Connection Pool Exhaustion" in result.output


def test_incident_report_json_stdout(seeded_repo: IncidentRepository):
    """Verify `devops-ai incident report --format json` emits valid JSON to stdout."""
    result = runner.invoke(app, ["incident", "report", "INC-20261124-0002", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["incident_id"] == "INC-20261124-0002"


def test_incident_report_to_file(tmp_path: Path, seeded_repo: IncidentRepository):
    """Verify `devops-ai incident report --output <file>` writes report to disk."""
    out_file = tmp_path / "postmortem.md"
    result = runner.invoke(
        app,
        [
            "incident",
            "report",
            "INC-20261124-0002",
            "--format",
            "markdown",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Post-mortem report exported successfully" in result.output
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# Incident Post-Mortem: INC-20261124-0002" in content


def test_incident_report_invalid_format(seeded_repo: IncidentRepository):
    """Verify unsupported report format returns exit code 2."""
    result = runner.invoke(app, ["incident", "report", "INC-20261124-0002", "--format", "docx"])
    assert result.exit_code == 2
    assert "Unsupported format: 'docx'" in result.output
