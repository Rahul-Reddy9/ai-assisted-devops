# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for observability and monitoring CLI subcommands."""

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from devops_ai.cli import app
from devops_ai.observability import (
    ContainerInfo,
    DockerMonitor,
    DockerStatus,
    K8sClusterStatus,
    K8sMonitor,
    K8sWorkloadInfo,
)

runner = CliRunner()


def test_cli_health_check_healthy():
    mock_liveness = {
        "status": "healthy",
        "service": "devops-ai",
        "version": "0.1.0",
        "uptime_seconds": 15.2,
        "timestamp": "2026-11-22T00:00:00Z",
    }
    mock_readiness = {
        "status": "ready",
        "checks": {
            "filesystem": {"status": "ok", "message": "Storage writable."},
            "memory": {"status": "ok", "message": "Memory healthy."},
            "provider": {"status": "ok", "message": "ollama endpoint reachable."},
        },
    }
    with (
        patch("devops_ai.cli.evaluate_liveness", return_value=mock_liveness),
        patch("devops_ai.cli.evaluate_readiness", return_value=(200, mock_readiness)),
    ):
        result = runner.invoke(app, ["health", "check"])
        assert result.exit_code == 0
        assert "READY" in result.output
        assert "Storage writable." in result.output
        assert "ollama" in result.output


def test_cli_health_check_json():
    mock_liveness = {"status": "healthy", "uptime_seconds": 10.0, "version": "0.1.0"}
    mock_readiness = {"status": "ready", "checks": {}}
    with (
        patch("devops_ai.cli.evaluate_liveness", return_value=mock_liveness),
        patch("devops_ai.cli.evaluate_readiness", return_value=(200, mock_readiness)),
    ):
        result = runner.invoke(app, ["health", "check", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["liveness"]["status"] == "healthy"
        assert data["readiness"]["status"] == "ready"


def test_cli_health_check_fatal_readiness():
    mock_liveness = {"status": "healthy", "uptime_seconds": 5.0, "version": "0.1.0"}
    mock_readiness = {
        "status": "unhealthy",
        "checks": {
            "filesystem": {"status": "failed", "message": "Read-only filesystem"},
            "memory": {"status": "ok", "message": "Healthy"},
        },
    }
    with (
        patch("devops_ai.cli.evaluate_liveness", return_value=mock_liveness),
        patch("devops_ai.cli.evaluate_readiness", return_value=(503, mock_readiness)),
    ):
        result = runner.invoke(app, ["health", "check"])
        assert result.exit_code == 1
        assert "UNHEALTHY" in result.output


def test_cli_metrics_dump():
    exposition_bytes = b"# HELP test_metric Test\n# TYPE test_metric counter\ntest_metric 42.0\n"
    with patch("devops_ai.cli.get_prometheus_exposition", return_value=exposition_bytes):
        result = runner.invoke(app, ["metrics", "dump"])
        assert result.exit_code == 0
        assert "test_metric 42.0" in result.output


def test_cli_metrics_serve_invalid_port():
    result = runner.invoke(app, ["metrics", "serve", "--port", "999999"])
    assert result.exit_code == 2
    assert "Invalid port" in result.output


def test_cli_monitor_docker_available():
    mock_status = DockerStatus(
        is_available=True,
        daemon_version="27.0.1",
        containers=[
            ContainerInfo(
                container_id="123456789abc",
                name="devops-worker",
                image="alpine:latest",
                status="Up 1 hour",
                state="running",
                created="2026-11-22",
                ports="8000/tcp",
            )
        ],
    )
    with patch.object(DockerMonitor, "get_status", return_value=mock_status):
        result = runner.invoke(app, ["monitor", "docker"])
        assert result.exit_code == 0
        assert "ONLINE" in result.output
        assert "123456789abc" in result.output
        assert "devops-work" in result.output


def test_cli_monitor_docker_unavailable():
    mock_status = DockerStatus(
        is_available=False,
        error="Docker daemon unreachable: socket not found",
    )
    with patch.object(DockerMonitor, "get_status", return_value=mock_status):
        result = runner.invoke(app, ["monitor", "docker"])
        assert result.exit_code == 1
        assert "UNAVAILABLE" in result.output
        assert "socket not found" in result.output


def test_cli_monitor_docker_json():
    mock_status = DockerStatus(
        is_available=True,
        daemon_version="27.0.1",
        containers=[],
    )
    with patch.object(DockerMonitor, "get_status", return_value=mock_status):
        result = runner.invoke(app, ["monitor", "docker", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["is_available"] is True
        assert data["daemon_version"] == "27.0.1"


def test_cli_monitor_k8s_available():
    mock_status = K8sClusterStatus(
        is_available=True,
        cluster_host="https://127.0.0.1:6443",
        workloads=[
            K8sWorkloadInfo(
                kind="Deployment",
                namespace="prod",
                name="api-gateway",
                status="3/3 Ready",
                age="2026-11-20T10:00:00Z",
            )
        ],
    )
    with patch.object(K8sMonitor, "get_status", return_value=mock_status):
        result = runner.invoke(app, ["monitor", "k8s", "-n", "prod"])
        assert result.exit_code == 0
        assert "CONNECTED" in result.output
        assert "api-gateway" in result.output
        assert "3/3 Ready" in result.output


def test_cli_monitor_k8s_unavailable():
    mock_status = K8sClusterStatus(
        is_available=False,
        error="Kubernetes cluster unreachable: connection refused",
    )
    with patch.object(K8sMonitor, "get_status", return_value=mock_status):
        result = runner.invoke(app, ["monitor", "k8s"])
        assert result.exit_code == 1
        assert "UNAVAILABLE" in result.output
        assert "connection refused" in result.output


def test_cli_log_analyze_success(tmp_path: Path):
    log_file = tmp_path / "test.log"
    log_content = (
        "2026-11-22T10:00:00Z [INFO] system started\n"
        "2026-11-22T10:00:05Z [ERROR] failed database connection\n"
        "2026-11-22T10:00:08Z [ERROR] retry connection failed\n"
        "2026-11-22T10:00:10Z [ERROR] fatal database timeout\n"
        "2026-11-22T10:00:12Z [ERROR] pool exhausted\n"
    )
    log_file.write_text(log_content, encoding="utf-8")

    result = runner.invoke(app, ["log-analyze", str(log_file), "--threshold", "3"])
    assert result.exit_code == 0
    assert "Log Analysis Results" in result.output
    assert "Total Parsed Entries: 5" in result.output


def test_cli_log_analyze_json(tmp_path: Path):
    log_file = tmp_path / "test.log"
    log_file.write_text("2026-11-22T10:00:00Z [INFO] test\n", encoding="utf-8")

    result = runner.invoke(app, ["log-analyze", str(log_file), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total_entries"] == 1
    assert "bursts" in data
    assert "anomalies" in data


def test_cli_log_analyze_json_fallback_when_sklearn_missing(tmp_path: Path):
    """Verify JSON output remains clean, valid machine-readable JSON even when scikit-learn is absent."""
    import sys

    log_file = tmp_path / "test.log"
    log_file.write_text("2026-11-22T10:00:00Z [INFO] test\n", encoding="utf-8")

    with patch.dict(sys.modules, {"sklearn": None, "sklearn.ensemble": None}):
        result = runner.invoke(app, ["log-analyze", str(log_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_entries"] == 1
        assert "bursts" in data
        assert "anomalies" in data
        assert data["detection_method"] == "statistical_percentile"


def test_cli_log_analyze_invalid_arguments(tmp_path: Path):
    log_file = tmp_path / "test.log"
    log_file.write_text("line\n", encoding="utf-8")

    res_window = runner.invoke(app, ["log-analyze", str(log_file), "--window", "-5"])
    assert res_window.exit_code == 2

    res_threshold = runner.invoke(app, ["log-analyze", str(log_file), "--threshold", "0"])
    assert res_threshold.exit_code == 2
