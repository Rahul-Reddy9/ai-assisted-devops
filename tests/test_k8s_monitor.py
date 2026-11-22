# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for read-only Kubernetes workload monitoring."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from devops_ai.observability.k8s_monitor import K8sMonitor
from devops_ai.observability.metrics import K8S_AVAILABLE_GAUGE


def test_k8s_check_availability_no_config():
    monitor = K8sMonitor()
    with patch.object(monitor, "_load_client", return_value=False):
        available = monitor.check_availability()
        assert available is False
        assert K8S_AVAILABLE_GAUGE._value.get() == 0.0


def test_k8s_check_availability_success():
    monitor = K8sMonitor()
    with (
        patch.object(monitor, "_load_client", return_value=True),
        patch("kubernetes.client.CoreV1Api.get_api_resources", return_value=MagicMock()),
    ):
        available = monitor.check_availability()
        assert available is True
        assert K8S_AVAILABLE_GAUGE._value.get() == 1.0


def test_k8s_check_availability_failure():
    monitor = K8sMonitor()
    with (
        patch.object(monitor, "_load_client", return_value=True),
        patch(
            "kubernetes.client.CoreV1Api.get_api_resources",
            side_effect=Exception("Connection refused"),
        ),
    ):
        available = monitor.check_availability()
        assert available is False
        assert K8S_AVAILABLE_GAUGE._value.get() == 0.0


def test_k8s_get_status_no_config():
    monitor = K8sMonitor()
    with patch.object(monitor, "_load_client", return_value=False):
        status = monitor.get_status()
        assert status.is_available is False
        assert "configuration not found" in (status.error or "")
        assert len(status.workloads) == 0


def test_k8s_get_status_success_with_workloads():
    monitor = K8sMonitor()
    now = datetime.now(timezone.utc)

    # Mock Deployment
    mock_dep = MagicMock()
    mock_dep.metadata.namespace = "default"
    mock_dep.metadata.name = "web-deployment"
    mock_dep.metadata.creation_timestamp = now
    mock_dep.status.ready_replicas = 2
    mock_dep.spec.replicas = 2

    # Mock Pod
    mock_pod = MagicMock()
    mock_pod.metadata.namespace = "default"
    mock_pod.metadata.name = "web-pod-xyz"
    mock_pod.metadata.creation_timestamp = now
    mock_pod.status.phase = "Running"

    # Mock Service
    mock_svc = MagicMock()
    mock_svc.metadata.namespace = "default"
    mock_svc.metadata.name = "web-service"
    mock_svc.metadata.creation_timestamp = now
    mock_svc.spec.type = "ClusterIP"
    mock_svc.spec.cluster_ip = "10.96.0.1"

    deps_res = MagicMock(items=[mock_dep])
    pods_res = MagicMock(items=[mock_pod])
    svcs_res = MagicMock(items=[mock_svc])

    with (
        patch.object(monitor, "_load_client", return_value=True),
        patch(
            "kubernetes.client.AppsV1Api.list_deployment_for_all_namespaces", return_value=deps_res
        ),
        patch("kubernetes.client.CoreV1Api.list_pod_for_all_namespaces", return_value=pods_res),
        patch("kubernetes.client.CoreV1Api.list_service_for_all_namespaces", return_value=svcs_res),
    ):
        status = monitor.get_status()
        assert status.is_available is True
        assert len(status.workloads) == 3

        kinds = {w.kind for w in status.workloads}
        assert kinds == {"Deployment", "Pod", "Service"}

        dep_info = next(w for w in status.workloads if w.kind == "Deployment")
        assert dep_info.name == "web-deployment"
        assert dep_info.status == "2/2 Ready"

        pod_info = next(w for w in status.workloads if w.kind == "Pod")
        assert pod_info.name == "web-pod-xyz"
        assert pod_info.status == "Running"

        svc_info = next(w for w in status.workloads if w.kind == "Service")
        assert svc_info.name == "web-service"
        assert "ClusterIP" in svc_info.status


def test_k8s_get_status_scrubs_secrets_on_exception():
    monitor = K8sMonitor()
    secret = "AIzaSyC-987654321098765432109876543210"

    with (
        patch.object(monitor, "_load_client", return_value=True),
        patch(
            "kubernetes.client.AppsV1Api.list_deployment_for_all_namespaces",
            side_effect=Exception(f"Unauthorized request token: Bearer {secret}"),
        ),
    ):
        # We need an unhandled outer exception to trigger the error branch
        # or have AppsV1Api constructor raise an exception
        pass

    with (
        patch.object(monitor, "_load_client", return_value=True),
        patch("kubernetes.client.AppsV1Api", side_effect=Exception(f"Failed with key {secret}")),
    ):
        status = monitor.get_status()
        assert status.is_available is False
        assert secret not in (status.error or "")
        assert "[REDACTED_API_KEY]" in (status.error or "")
