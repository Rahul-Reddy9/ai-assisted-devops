# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for incident reporting formats (Rich terminal, Markdown, and JSON)."""

import datetime
import json
from pathlib import Path

import pytest
from rich.console import Console

from devops_ai.core.exceptions import DevOpsAIError
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
from devops_ai.incident.reporter import IncidentReporter, sanitize_text


@pytest.fixture
def sample_incident() -> Incident:
    """Create a fully populated incident with root cause, recommendations, and evidence."""
    now = datetime.datetime.now(datetime.timezone.utc)
    earlier = now - datetime.timedelta(minutes=10)

    alerts = [
        Alert(
            alert_id="ALT-101",
            source=AlertSource.LOGS,
            component="checkout-service",
            message="java.lang.OutOfMemoryError: Java heap space",
            severity=Severity.CRITICAL,
            timestamp=earlier,
            fingerprint="fp-101",
        ),
        Alert(
            alert_id="ALT-102",
            source=AlertSource.DOCKER,
            component="checkout-service",
            message="Container exited with status 137 (OOMKilled: true)",
            severity=Severity.CRITICAL,
            timestamp=earlier + datetime.timedelta(seconds=15),
            fingerprint="fp-102",
        ),
    ]

    evidence = CorrelatedEvidence(
        evidence_catalog={
            "EVID-LOG-01": "java.lang.OutOfMemoryError: Java heap space in checkout-service",
            "EVID-CONTAINER-01": "Container exit code 137 (OOMKilled: True)",
            "EVID-METRIC-01": "Memory utilization at 98.5%",
        },
        log_snippets=["2026-11-23T10:00:00Z ERROR OutOfMemoryError in thread main"],
        metric_anomalies={"container_memory_usage_ratio": 0.985},
    )

    root_cause = RootCauseHypothesis(
        title="Container OOMKilled (Out of Memory)",
        description="The container process was terminated by the Linux OOM killer after exceeding allocated memory limit.",
        confidence=0.95,
        tier="tier_1_deterministic",
        evidence_refs=["EVID-LOG-01", "EVID-CONTAINER-01"],
        supporting_evidence=[
            "Container OOMKilled exit code 137",
            "Explicit OutOfMemoryError in logs",
        ],
        contradicting_evidence=[],
        is_confirmed=True,
        is_unsupported=False,
    )

    alternative = RootCauseHypothesis(
        title="Memory Leak in Application Process",
        description="Monotonic heap growth leading to exhaustion.",
        confidence=0.72,
        tier="tier_1_deterministic",
        evidence_refs=["EVID-LOG-01"],
        supporting_evidence=["Heap space error"],
        contradicting_evidence=[],
        is_confirmed=False,
        is_unsupported=False,
    )

    recommendations = [
        RemediationAction(
            action_id="ACT-001",
            title="Inspect Container Exit Reason and Memory Limit",
            description="Inspect Docker container state to confirm OOMKilled termination flag.",
            action_type="diagnostic",
            risk_level=RiskLevel.SAFE,
            command="docker inspect checkout-service --format '{{.State.OOMKilled}}'",
            verification_step="Verify command returns 'true'.",
        ),
        RemediationAction(
            action_id="ACT-002",
            title="Increase Pod / Container Memory Limit",
            description="Increase memory resource limit from 512Mi to 1Gi to prevent heap exhaustion.",
            action_type="permanent_fix",
            risk_level=RiskLevel.MODERATE,
            command="kubectl set resources deployment checkout-service --limits=memory=1Gi",
            verification_step="Monitor container restart count and memory metrics for 10 minutes.",
            rollback_guidance="kubectl rollout undo deployment checkout-service",
        ),
    ]

    timeline = [
        TimelineEntry(
            timestamp=earlier,
            source=AlertSource.LOGS,
            component="checkout-service",
            event="Error burst detected in log stream",
            severity=Severity.HIGH,
        ),
        TimelineEntry(
            timestamp=earlier + datetime.timedelta(seconds=15),
            source=AlertSource.DOCKER,
            component="checkout-service",
            event="Container terminated with status 137",
            severity=Severity.CRITICAL,
        ),
    ]

    return Incident(
        incident_id="INC-20261124-TEST1",
        title="Critical Memory Saturation and OOMKill on checkout-service",
        category=Category.CONTAINER,
        severity=Severity.CRITICAL,
        status=IncidentStatus.ANALYZED,
        created_at=earlier,
        updated_at=now,
        components_affected=["checkout-service"],
        alerts=alerts,
        timeline=timeline,
        evidence=evidence,
        root_cause=root_cause,
        alternative_hypotheses=[alternative],
        recommendations=recommendations,
    )


@pytest.fixture
def minimal_incident() -> Incident:
    """Create a minimal incident with no root cause or recommendations."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return Incident(
        incident_id="INC-20261124-MIN1",
        title="Uncorrelated Warning Event",
        category=Category.UNKNOWN,
        severity=Severity.LOW,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
        components_affected=["gateway"],
    )


def test_render_terminal_complete_incident(sample_incident: Incident):
    """Test terminal rendering of a fully populated incident with Rich console."""
    test_console = Console(record=True, width=120)
    IncidentReporter.render_terminal(sample_incident, console=test_console)
    output = test_console.export_text()

    assert "INC-20261124-TEST1" in output
    assert "checkout-service" in output
    assert "CRITICAL" in output
    assert "ANALYZED" in output
    assert "Container OOMKilled" in output
    assert "95%" in output
    assert "CONFIRMED" in output
    assert "ADVISORY ONLY" in output
    assert "ACT-001" in output
    assert "SAFE" in output
    assert "MODERATE" in output
    assert "docker inspect" in output
    assert "Incident Timeline" in output


def test_render_terminal_minimal_incident(minimal_incident: Incident):
    """Test terminal rendering of a minimal incident without crashing."""
    test_console = Console(record=True, width=120)
    IncidentReporter.render_terminal(minimal_incident, console=test_console)
    output = test_console.export_text()

    assert "INC-20261124-MIN1" in output
    assert "gateway" in output
    assert "LOW" in output
    assert "DETECTED" in output
    assert "No root cause analysis performed yet" in output


def test_render_markdown_complete_incident(sample_incident: Incident):
    """Test Markdown post-mortem generation for a complete incident."""
    md = IncidentReporter.render_markdown(sample_incident)

    assert "# Incident Post-Mortem: INC-20261124-TEST1" in md
    assert "## Executive Summary" in md
    assert "| **Incident ID** | `INC-20261124-TEST1` |" in md
    assert "| **Status** | `ANALYZED` |" in md
    assert "| **Severity** | `CRITICAL` |" in md
    assert "| **Category** | `CONTAINER` |" in md
    assert "## Primary Root Cause Analysis" in md
    assert "### Container OOMKilled (Out of Memory)" in md
    assert "**Confidence Score:** `95%`" in md
    assert "**Verification Status:** `Confirmed`" in md
    assert "EVID-CONTAINER-01" in md
    assert "## Alternative Hypotheses Considered" in md
    assert "Memory Leak in Application Process" in md
    assert "## Remediation Advisory" in md
    assert "> [!WARNING]" in md
    assert "ADVISORY ONLY" in md
    assert "`ACT-001`" in md
    assert "`ACT-002`" in md
    assert "## Incident Timeline" in md
    assert "## Correlated Evidence Catalog" in md


def test_render_markdown_minimal_incident(minimal_incident: Incident):
    """Test Markdown post-mortem generation for a minimal incident."""
    md = IncidentReporter.render_markdown(minimal_incident)

    assert "# Incident Post-Mortem: INC-20261124-MIN1" in md
    assert "No root cause analysis performed yet" in md
    assert "No remediation recommendations generated yet" in md


def test_render_json_validity(sample_incident: Incident):
    """Test JSON rendering outputs valid parseable JSON deserializable back to Incident."""
    json_str = IncidentReporter.render_json(sample_incident)
    data = json.loads(json_str)

    assert data["incident_id"] == "INC-20261124-TEST1"
    assert data["severity"] == "CRITICAL"
    assert data["status"] == "ANALYZED"
    assert data["root_cause"]["confidence"] == 0.95

    # Verify deserialization
    roundtrip = Incident.model_validate(data)
    assert roundtrip.incident_id == sample_incident.incident_id
    assert roundtrip.root_cause.title == sample_incident.root_cause.title


def test_export_report_markdown_to_file(tmp_path: Path, sample_incident: Incident):
    """Test exporting Markdown report to a destination file."""
    output_file = tmp_path / "post_mortem.md"
    content = IncidentReporter.export_report(
        sample_incident, format="markdown", output_path=output_file
    )

    assert output_file.exists()
    file_text = output_file.read_text(encoding="utf-8")
    assert file_text == content
    assert "# Incident Post-Mortem" in file_text


def test_export_report_json_to_file(tmp_path: Path, sample_incident: Incident):
    """Test exporting JSON report to a destination file."""
    output_file = tmp_path / "incident.json"
    content = IncidentReporter.export_report(
        sample_incident, format="json", output_path=output_file
    )

    assert output_file.exists()
    file_text = output_file.read_text(encoding="utf-8")
    assert file_text == content
    loaded = json.loads(file_text)
    assert loaded["incident_id"] == "INC-20261124-TEST1"


def test_export_report_terminal_to_file(tmp_path: Path, sample_incident: Incident):
    """Test exporting terminal text representation to a destination file."""
    output_file = tmp_path / "terminal.txt"
    content = IncidentReporter.export_report(
        sample_incident, format="terminal", output_path=output_file
    )

    assert output_file.exists()
    assert len(content) > 0
    assert "INC-20261124-TEST1" in content


def test_export_report_unsupported_format(sample_incident: Incident):
    """Test exporting with an unsupported format raises DevOpsAIError."""
    with pytest.raises(DevOpsAIError) as exc_info:
        IncidentReporter.export_report(sample_incident, format="pdf")
    assert "Unsupported report format: 'pdf'" in str(exc_info.value)


def test_secret_redaction_in_reports():
    """Test that Google API keys are scrubbed from text and JSON outputs."""
    raw_text = "Encountered error with key AIzaSyA1234567890abcdefghijklmnopqrstuv in request"
    cleaned = sanitize_text(raw_text)
    assert "AIzaSy" not in cleaned
    assert "[REDACTED_API_KEY]" in cleaned
