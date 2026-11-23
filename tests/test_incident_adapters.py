# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying monitoring ingestion adapters for Phase 3 components."""

import datetime
from devops_ai.incident.adapters import (
    DockerMonitorAdapter,
    HealthProbeAdapter,
    K8sMonitorAdapter,
    LogIngestionAdapter,
    MetricsAdapter,
)
from devops_ai.incident.models import AlertSource, Severity
from devops_ai.observability.docker_monitor import ContainerInfo, DockerStatus
from devops_ai.observability.k8s_monitor import K8sClusterStatus, K8sWorkloadInfo
from devops_ai.observability.log_analyzer import BurstDetectionResult


def test_log_ingestion_adapter_burst():
    """Verify LogIngestionAdapter converts BurstDetectionResult into Alert."""
    now = datetime.datetime.now(datetime.timezone.utc)
    burst = BurstDetectionResult(
        has_burst=True,
        error_count=8,
        window_start=now,
        window_end=now + datetime.timedelta(seconds=30),
        message="Detected 8 error logs within 30s",
    )

    alert = LogIngestionAdapter.adapt_burst(burst, component="checkout-service")
    assert alert.source == AlertSource.LOGS
    assert alert.severity == Severity.HIGH
    assert alert.component == "checkout-service"
    assert "8 error logs" in alert.message
    assert alert.raw_data["error_count"] == 8


def test_log_ingestion_adapter_anomaly():
    """Verify LogIngestionAdapter converts anomaly dict into Alert."""
    anomaly_data = {"message": "Sudden volume divergence", "score": -0.42}
    alert = LogIngestionAdapter.adapt_anomaly(anomaly_data, component="api-gateway")

    assert alert.source == AlertSource.LOGS
    assert alert.severity == Severity.MEDIUM
    assert alert.component == "api-gateway"
    assert "Sudden volume divergence" in alert.message


def test_health_probe_adapter():
    """Verify HealthProbeAdapter converts HTTP 503 and degraded states into Alerts."""
    # 1. Fatal readiness failure (503)
    payload_fatal = {
        "status": "unhealthy",
        "checks": {
            "filesystem": {"status": "error", "message": "Read-only filesystem"},
            "memory": {"status": "ok"},
        },
    }
    alert_fatal = HealthProbeAdapter.adapt_readiness_result(503, payload_fatal)
    assert alert_fatal is not None
    assert alert_fatal.source == AlertSource.HEALTH_PROBE
    assert alert_fatal.severity == Severity.CRITICAL
    assert "Readiness probe failed" in alert_fatal.message
    assert "filesystem" in alert_fatal.message

    # 2. Degraded readiness (200 OK with warnings)
    payload_degraded = {
        "status": "degraded",
        "checks": {
            "temp_storage": {"status": "degraded", "message": "Low free space"},
        },
    }
    alert_degraded = HealthProbeAdapter.adapt_readiness_result(200, payload_degraded)
    assert alert_degraded is not None
    assert alert_degraded.severity == Severity.MEDIUM

    # 3. Healthy probe (200 OK without degraded checks)
    payload_healthy = {
        "status": "healthy",
        "checks": {"filesystem": {"status": "ok"}},
    }
    assert HealthProbeAdapter.adapt_readiness_result(200, payload_healthy) is None


def test_docker_monitor_adapter_and_offline():
    """Verify DockerMonitorAdapter extracts exited containers and handles offline daemon."""
    # 1. Offline Docker daemon -> graceful empty list
    offline_status = DockerStatus(is_available=False, error="Docker daemon offline")
    assert DockerMonitorAdapter.adapt_status(offline_status) == []

    # 2. Online Docker with exited and running containers
    online_status = DockerStatus(
        is_available=True,
        daemon_version="24.0.7",
        containers=[
            ContainerInfo(
                container_id="c1",
                name="app-web",
                image="nginx:latest",
                status="Up 2 hours",
                state="running",
                created="2h ago",
                ports="80/tcp",
            ),
            ContainerInfo(
                container_id="c2",
                name="app-db",
                image="postgres:15",
                status="Exited (137) 5 minutes ago",
                state="exited",
                created="1d ago",
                ports="",
            ),
        ],
    )

    alerts = DockerMonitorAdapter.adapt_status(online_status)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.source == AlertSource.DOCKER
    assert alert.component == "app-db"
    assert alert.severity == Severity.HIGH
    assert "Exited (137)" in alert.message


def test_k8s_monitor_adapter_and_offline():
    """Verify K8sMonitorAdapter extracts CrashLoopBackOff/ImagePullBackOff and handles offline cluster."""
    # 1. Offline Kubernetes -> graceful empty list
    offline_status = K8sClusterStatus(is_available=False, error="Cluster unreachable")
    assert K8sMonitorAdapter.adapt_status(offline_status) == []

    # 2. Workloads with CrashLoopBackOff and ImagePullBackOff
    online_status = K8sClusterStatus(
        is_available=True,
        workloads=[
            K8sWorkloadInfo(
                kind="Pod",
                namespace="prod",
                name="payment-processor-abc",
                status="CrashLoopBackOff",
                age="15m",
            ),
            K8sWorkloadInfo(
                kind="Pod",
                namespace="prod",
                name="inventory-service-xyz",
                status="ImagePullBackOff",
                age="5m",
            ),
            K8sWorkloadInfo(
                kind="Pod",
                namespace="prod",
                name="frontend-123",
                status="Running",
                age="2d",
            ),
        ],
    )

    alerts = K8sMonitorAdapter.adapt_status(online_status)
    assert len(alerts) == 2

    crash_alert = next(a for a in alerts if "payment-processor" in a.component)
    assert crash_alert.severity == Severity.HIGH
    assert "CrashLoopBackOff" in crash_alert.message

    image_alert = next(a for a in alerts if "inventory-service" in a.component)
    assert image_alert.severity == Severity.MEDIUM
    assert "ImagePullBackOff" in image_alert.message


def test_metrics_adapter():
    """Verify MetricsAdapter alerts on CPU and memory thresholds."""
    # 1. CPU saturation >= 90%
    cpu_alerts = MetricsAdapter.adapt_system_metrics(cpu_percent=94.5, memory_used_bytes=1000)
    assert len(cpu_alerts) == 1
    assert cpu_alerts[0].source == AlertSource.METRICS
    assert cpu_alerts[0].severity == Severity.HIGH
    assert "94.5%" in cpu_alerts[0].message

    # 2. Memory saturation >= 85%
    # 8.8 GB used of 10 GB total = 88%
    mem_alerts = MetricsAdapter.adapt_system_metrics(
        cpu_percent=10.0,
        memory_used_bytes=8800000000,
        memory_total_bytes=10000000000,
    )
    assert len(mem_alerts) == 1
    assert mem_alerts[0].severity == Severity.MEDIUM
    assert "88.0%" in mem_alerts[0].message

    # 3. Normal metrics -> no alerts
    normal_alerts = MetricsAdapter.adapt_system_metrics(
        cpu_percent=30.0,
        memory_used_bytes=4000000000,
        memory_total_bytes=10000000000,
    )
    assert len(normal_alerts) == 0
