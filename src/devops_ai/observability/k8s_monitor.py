# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Read-only Kubernetes cluster and workload monitoring with graceful degradation."""

from dataclasses import dataclass, field
import datetime
from typing import List, Optional

from devops_ai.core.logging import _scrub_secrets, get_logger
from devops_ai.observability.metrics import record_k8s_status

logger = get_logger("observability.k8s_monitor")


@dataclass(frozen=True)
class K8sWorkloadInfo:
    """Read-only snapshot of a Kubernetes workload resource."""

    kind: str
    namespace: str
    name: str
    status: str
    age: str


@dataclass(frozen=True)
class K8sClusterStatus:
    """Consolidated Kubernetes cluster connectivity and workload inventory."""

    is_available: bool
    cluster_host: Optional[str] = None
    workloads: List[K8sWorkloadInfo] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class K8sMonitor:
    """Provides safe, read-only inspection of Kubernetes workloads."""

    def __init__(
        self,
        kubeconfig_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> None:
        self.kubeconfig_path = kubeconfig_path
        self.context = context
        self._config_loaded = False
        self._api_client = None

    def _load_client(self) -> bool:
        """Attempt to configure Kubernetes client safely."""
        if self._config_loaded:
            return True

        try:
            from kubernetes import client, config  # type: ignore[import-untyped]

            try:
                config.load_incluster_config()
                self._config_loaded = True
                self._api_client = client.ApiClient()
                return True
            except config.ConfigException:
                pass

            config.load_kube_config(
                config_file=self.kubeconfig_path,
                context=self.context,
            )
            self._config_loaded = True
            self._api_client = client.ApiClient()
            return True
        except Exception as exc:
            logger.debug(f"Could not load Kubernetes configuration: {exc}")
            return False

    def check_availability(self, timeout: float = 3.0) -> bool:
        """Probe Kubernetes cluster API server connectivity."""
        if not self._load_client():
            record_k8s_status(False)
            return False

        try:
            from kubernetes import client  # type: ignore[import-untyped]

            core_api = client.CoreV1Api(api_client=self._api_client)
            # Query namespace list or version with strict timeout
            core_api.get_api_resources(_request_timeout=timeout)
            record_k8s_status(True)
            return True
        except Exception as exc:
            logger.debug(f"Kubernetes cluster probe failed: {exc}")
            record_k8s_status(False)
            return False

    def get_status(
        self,
        namespace: Optional[str] = None,
        timeout: float = 5.0,
    ) -> K8sClusterStatus:
        """Safely list Deployments, Pods, and Services without mutating state."""
        if not self._load_client():
            record_k8s_status(False)
            return K8sClusterStatus(
                is_available=False,
                error="Kubernetes configuration not found or invalid.",
            )

        try:
            from kubernetes import client  # type: ignore[import-untyped]

            cluster_host = None
            if self._api_client and hasattr(self._api_client, "configuration"):
                cluster_host = getattr(self._api_client.configuration, "host", None)

            apps_api = client.AppsV1Api(api_client=self._api_client)
            core_api = client.CoreV1Api(api_client=self._api_client)

            workloads: List[K8sWorkloadInfo] = []
            query_errors: List[str] = []

            # 1. Fetch Deployments
            try:
                if namespace:
                    deps = apps_api.list_namespaced_deployment(
                        namespace=namespace, _request_timeout=timeout
                    )
                else:
                    deps = apps_api.list_deployment_for_all_namespaces(_request_timeout=timeout)

                for dep in deps.items:
                    ready = dep.status.ready_replicas or 0
                    spec_replicas = dep.spec.replicas or 0
                    created = (
                        dep.metadata.creation_timestamp.isoformat()
                        if dep.metadata.creation_timestamp
                        else ""
                    )
                    workloads.append(
                        K8sWorkloadInfo(
                            kind="Deployment",
                            namespace=dep.metadata.namespace or "default",
                            name=dep.metadata.name or "",
                            status=f"{ready}/{spec_replicas} Ready",
                            age=created,
                        )
                    )
            except Exception as dep_err:
                query_errors.append(str(dep_err))
                logger.debug(f"Could not list deployments: {dep_err}")

            # 2. Fetch Pods
            try:
                if namespace:
                    pods = core_api.list_namespaced_pod(
                        namespace=namespace, _request_timeout=timeout
                    )
                else:
                    pods = core_api.list_pod_for_all_namespaces(_request_timeout=timeout)

                for pod in pods.items:
                    phase = pod.status.phase or "Unknown"
                    created = (
                        pod.metadata.creation_timestamp.isoformat()
                        if pod.metadata.creation_timestamp
                        else ""
                    )
                    workloads.append(
                        K8sWorkloadInfo(
                            kind="Pod",
                            namespace=pod.metadata.namespace or "default",
                            name=pod.metadata.name or "",
                            status=phase,
                            age=created,
                        )
                    )
            except Exception as pod_err:
                query_errors.append(str(pod_err))
                logger.debug(f"Could not list pods: {pod_err}")

            # 3. Fetch Services
            try:
                if namespace:
                    svcs = core_api.list_namespaced_service(
                        namespace=namespace, _request_timeout=timeout
                    )
                else:
                    svcs = core_api.list_service_for_all_namespaces(_request_timeout=timeout)

                for svc in svcs.items:
                    svc_type = svc.spec.type or "ClusterIP"
                    cluster_ip = svc.spec.cluster_ip or ""
                    created = (
                        svc.metadata.creation_timestamp.isoformat()
                        if svc.metadata.creation_timestamp
                        else ""
                    )
                    workloads.append(
                        K8sWorkloadInfo(
                            kind="Service",
                            namespace=svc.metadata.namespace or "default",
                            name=svc.metadata.name or "",
                            status=f"{svc_type} ({cluster_ip})",
                            age=created,
                        )
                    )
            except Exception as svc_err:
                query_errors.append(str(svc_err))
                logger.debug(f"Could not list services: {svc_err}")

            if len(query_errors) == 3:
                safe_err = _scrub_secrets(query_errors[0])
                record_k8s_status(False)
                return K8sClusterStatus(
                    is_available=False,
                    cluster_host=cluster_host,
                    error=f"Kubernetes cluster queries failed: {safe_err}",
                )

            record_k8s_status(True)
            return K8sClusterStatus(
                is_available=True,
                cluster_host=cluster_host,
                workloads=workloads,
            )

        except Exception as exc:
            record_k8s_status(False)
            safe_err = _scrub_secrets(str(exc))
            return K8sClusterStatus(
                is_available=False,
                error=f"Kubernetes cluster unreachable: {safe_err}",
            )
