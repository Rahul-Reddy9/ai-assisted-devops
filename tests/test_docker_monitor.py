# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for read-only Docker container monitoring."""

import subprocess
from unittest.mock import MagicMock, patch

from devops_ai.observability.docker_monitor import DockerMonitor
from devops_ai.observability.metrics import DOCKER_AVAILABLE_GAUGE


def test_docker_check_availability_success():
    monitor = DockerMonitor()
    mock_res = MagicMock(returncode=0, stdout="27.0.1\n", stderr="")
    with patch("subprocess.run", return_value=mock_res):
        available = monitor.check_availability()
        assert available is True
        assert DOCKER_AVAILABLE_GAUGE._value.get() == 1.0


def test_docker_check_availability_failure():
    monitor = DockerMonitor()
    mock_res = MagicMock(returncode=1, stdout="", stderr="daemon not running")
    with patch("subprocess.run", return_value=mock_res):
        available = monitor.check_availability()
        assert available is False
        assert DOCKER_AVAILABLE_GAUGE._value.get() == 0.0


def test_docker_check_availability_file_not_found():
    monitor = DockerMonitor()
    with patch("subprocess.run", side_effect=FileNotFoundError("Executable not found")):
        available = monitor.check_availability()
        assert available is False
        assert DOCKER_AVAILABLE_GAUGE._value.get() == 0.0


def test_docker_get_status_success():
    monitor = DockerMonitor()
    ver_mock = MagicMock(returncode=0, stdout="27.0.1", stderr="")
    ps_json = (
        '{"ID":"abcdef1234567890","Names":"devops-web","Image":"python:3.13-slim",'
        '"Status":"Up 3 hours","State":"running","CreatedAt":"2026-11-22 10:00:00",'
        '"Ports":"0.0.0.0:8080->8080/tcp"}\n'
    )
    ps_mock = MagicMock(returncode=0, stdout=ps_json, stderr="")

    with patch("subprocess.run", side_effect=[ver_mock, ps_mock]):
        status = monitor.get_status(all_containers=True)
        assert status.is_available is True
        assert status.daemon_version == "27.0.1"
        assert len(status.containers) == 1
        container = status.containers[0]
        assert container.container_id == "abcdef123456"
        assert container.name == "devops-web"
        assert container.image == "python:3.13-slim"
        assert container.state == "running"
        assert container.ports == "0.0.0.0:8080->8080/tcp"


def test_docker_get_status_daemon_down():
    monitor = DockerMonitor()
    ver_mock = MagicMock(returncode=1, stdout="", stderr="open //pipe/docker: not found")
    with patch("subprocess.run", return_value=ver_mock):
        status = monitor.get_status()
        assert status.is_available is False
        assert status.daemon_version is None
        assert "Docker daemon unreachable" in (status.error or "")
        assert len(status.containers) == 0


def test_docker_get_status_ps_timeout():
    monitor = DockerMonitor()
    ver_mock = MagicMock(returncode=0, stdout="27.0.1", stderr="")
    with patch(
        "subprocess.run",
        side_effect=[ver_mock, subprocess.TimeoutExpired(cmd="docker ps", timeout=5.0)],
    ):
        status = monitor.get_status()
        assert status.is_available is False
        assert "timed out" in (status.error or "")


def test_docker_get_status_scrubs_secrets():
    monitor = DockerMonitor()
    secret = "AIzaSyD-123456789012345678901234567890"
    ver_mock = MagicMock(returncode=1, stdout="", stderr=f"Connection refused with key={secret}")
    with patch("subprocess.run", return_value=ver_mock):
        status = monitor.get_status()
        assert status.is_available is False
        assert secret not in (status.error or "")
        assert "[REDACTED_API_KEY]" in (status.error or "")
