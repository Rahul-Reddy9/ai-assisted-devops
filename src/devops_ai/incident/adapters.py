# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Monitoring ingestion adapters bridging Phase 3 telemetry into Phase 4 alerts."""

import datetime
import hashlib
from typing import Any, Dict, List, Optional
import uuid

from devops_ai.incident.models import Alert, AlertSource, Severity
from devops_ai.observability.docker_monitor import DockerStatus
from devops_ai.observability.k8s_monitor import K8sClusterStatus
from devops_ai.observability.log_analyzer import BurstDetectionResult


def _generate_alert_id() -> str:
    """Generate unique identifier for an ingested alert."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    rand_suffix = uuid.uuid4().hex[:6].upper()
    return f"ALT-{ts}-{rand_suffix}"


def _compute_fingerprint(source: AlertSource, component: str, pattern: str) -> str:
    """Calculate deterministic SHA-256 fingerprint for deduplication."""
    norm_comp = component.strip().lower()
    norm_pat = pattern.strip().lower()
    raw = f"{source.value}:{norm_comp}:{norm_pat}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LogIngestionAdapter:
    """Adapts Phase 3 LogAnalyzer outputs into standardized incident alerts."""

    @staticmethod
    def adapt_burst(
        burst: BurstDetectionResult,
        component: str = "application",
    ) -> Alert:
        """Convert a detected error burst into a HIGH severity Alert."""
        now = datetime.datetime.now(datetime.timezone.utc)
        fingerprint = _compute_fingerprint(AlertSource.LOGS, component, "error_burst_spike")
        return Alert(
            alert_id=_generate_alert_id(),
            source=AlertSource.LOGS,
            timestamp=burst.window_start or now,
            component=component,
            message=burst.message,
            severity=Severity.HIGH,
            fingerprint=fingerprint,
            recurrence_count=1,
            raw_data={
                "error_count": burst.error_count,
                "window_start": str(burst.window_start),
                "window_end": str(burst.window_end),
            },
        )

    @staticmethod
    def adapt_anomaly(
        anomaly: Dict[str, Any],
        component: str = "application",
    ) -> Alert:
        """Convert a machine learning or statistical anomaly into a MEDIUM severity Alert."""
        now = datetime.datetime.now(datetime.timezone.utc)
        pattern = str(anomaly.get("message") or "statistical_outlier")[:50]
        fingerprint = _compute_fingerprint(AlertSource.LOGS, component, f"anomaly:{pattern}")
        return Alert(
            alert_id=_generate_alert_id(),
            source=AlertSource.LOGS,
            timestamp=now,
            component=component,
            message=f"Statistical log anomaly detected: {anomaly.get('message', 'Unusual log volume/pattern')}",
            severity=Severity.MEDIUM,
            fingerprint=fingerprint,
            recurrence_count=1,
            raw_data=anomaly,
        )


class HealthProbeAdapter:
    """Adapts Phase 3 liveness and readiness probe evaluations into incident alerts."""

    @staticmethod
    def adapt_readiness_result(
        status_code: int,
        payload: Dict[str, Any],
        component: str = "devops-ai",
    ) -> Optional[Alert]:
        """Convert failed or degraded readiness probes into an Alert."""
        now = datetime.datetime.now(datetime.timezone.utc)
        checks = payload.get("checks", {})
        failed_checks = [k for k, v in checks.items() if v.get("status") != "ok"]

        if status_code == 503:
            pattern = f"readiness_fatal:{','.join(sorted(failed_checks)) or 'probe_failed'}"
            fingerprint = _compute_fingerprint(AlertSource.HEALTH_PROBE, component, pattern)
            return Alert(
                alert_id=_generate_alert_id(),
                source=AlertSource.HEALTH_PROBE,
                timestamp=now,
                component=component,
                message=f"Readiness probe failed (HTTP 503). Failed checks: {', '.join(failed_checks) or 'unknown'}",
                severity=Severity.CRITICAL,
                fingerprint=fingerprint,
                recurrence_count=1,
                raw_data=payload,
            )
        elif payload.get("status") == "degraded" or failed_checks:
            pattern = f"readiness_degraded:{','.join(sorted(failed_checks))}"
            fingerprint = _compute_fingerprint(AlertSource.HEALTH_PROBE, component, pattern)
            return Alert(
                alert_id=_generate_alert_id(),
                source=AlertSource.HEALTH_PROBE,
                timestamp=now,
                component=component,
                message=f"Service operational but degraded: {', '.join(failed_checks)}",
                severity=Severity.MEDIUM,
                fingerprint=fingerprint,
                recurrence_count=1,
                raw_data=payload,
            )

        return None


class DockerMonitorAdapter:
    """Adapts Phase 3 Docker container snapshots into incident alerts."""

    @staticmethod
    def adapt_status(status: DockerStatus) -> List[Alert]:
        """Extract non-running, exited, or dead container states as alerts."""
        if not status.is_available:
            return []

        now = datetime.datetime.now(datetime.timezone.utc)
        alerts: List[Alert] = []

        for container in status.containers:
            state = container.state.lower().strip()
            # Non-running states require alerting
            if state in ("exited", "dead"):
                # Determine severity based on exit status
                st_lower = container.status.lower()
                is_error_exit = (
                    ("exit (" in st_lower or "exited (" in st_lower)
                    and "exit (0)" not in st_lower
                    and "exited (0)" not in st_lower
                )
                severity = Severity.HIGH if is_error_exit else Severity.MEDIUM
                pattern = f"container_state:{state}:{container.status}"
                fingerprint = _compute_fingerprint(AlertSource.DOCKER, container.name, pattern)

                alerts.append(
                    Alert(
                        alert_id=_generate_alert_id(),
                        source=AlertSource.DOCKER,
                        timestamp=now,
                        component=container.name,
                        message=f"Container '{container.name}' is non-running ({state}). Status: {container.status}",
                        severity=severity,
                        fingerprint=fingerprint,
                        recurrence_count=1,
                        raw_data={
                            "container_id": container.container_id,
                            "name": container.name,
                            "image": container.image,
                            "state": container.state,
                            "status": container.status,
                        },
                    )
                )

        return alerts


class K8sMonitorAdapter:
    """Adapts Phase 3 Kubernetes workload status snapshots into incident alerts."""

    @staticmethod
    def adapt_status(status: K8sClusterStatus) -> List[Alert]:
        """Identify crash loops, image pull failures, or unready pods as alerts."""
        if not status.is_available:
            return []

        now = datetime.datetime.now(datetime.timezone.utc)
        alerts: List[Alert] = []

        for workload in status.workloads:
            st = workload.status.strip()
            st_upper = st.upper()

            if "CRASHLOOPBACKOFF" in st_upper or "ERROR" in st_upper:
                comp = f"{workload.namespace}/{workload.name}"
                fingerprint = _compute_fingerprint(AlertSource.KUBERNETES, comp, f"k8s_crash:{st}")
                alerts.append(
                    Alert(
                        alert_id=_generate_alert_id(),
                        source=AlertSource.KUBERNETES,
                        timestamp=now,
                        component=comp,
                        message=f"Kubernetes workload '{workload.name}' in namespace '{workload.namespace}' is failing: {st}",
                        severity=Severity.HIGH,
                        fingerprint=fingerprint,
                        recurrence_count=1,
                        raw_data={
                            "kind": workload.kind,
                            "namespace": workload.namespace,
                            "name": workload.name,
                            "status": workload.status,
                            "age": workload.age,
                        },
                    )
                )
            elif (
                "IMAGEPULLBACKOFF" in st_upper
                or "ERRIMAGEPULL" in st_upper
                or "PENDING" in st_upper
            ):
                comp = f"{workload.namespace}/{workload.name}"
                fingerprint = _compute_fingerprint(
                    AlertSource.KUBERNETES, comp, f"k8s_image_pending:{st}"
                )
                alerts.append(
                    Alert(
                        alert_id=_generate_alert_id(),
                        source=AlertSource.KUBERNETES,
                        timestamp=now,
                        component=comp,
                        message=f"Kubernetes workload '{workload.name}' in namespace '{workload.namespace}' has deployment issue: {st}",
                        severity=Severity.MEDIUM,
                        fingerprint=fingerprint,
                        recurrence_count=1,
                        raw_data={
                            "kind": workload.kind,
                            "namespace": workload.namespace,
                            "name": workload.name,
                            "status": workload.status,
                            "age": workload.age,
                        },
                    )
                )

        return alerts


class MetricsAdapter:
    """Adapts host system and Prometheus metric thresholds into incident alerts."""

    @staticmethod
    def adapt_system_metrics(
        cpu_percent: float,
        memory_used_bytes: float,
        memory_total_bytes: Optional[float] = None,
        component: str = "host_system",
    ) -> List[Alert]:
        """Check CPU and memory saturation metrics against alert thresholds."""
        now = datetime.datetime.now(datetime.timezone.utc)
        alerts: List[Alert] = []

        if cpu_percent >= 90.0:
            fingerprint = _compute_fingerprint(AlertSource.METRICS, component, "cpu_saturation_90")
            alerts.append(
                Alert(
                    alert_id=_generate_alert_id(),
                    source=AlertSource.METRICS,
                    timestamp=now,
                    component=component,
                    message=f"Host CPU saturation detected: {cpu_percent:.1f}% >= 90.0% threshold",
                    severity=Severity.HIGH,
                    fingerprint=fingerprint,
                    recurrence_count=1,
                    raw_data={"cpu_percent": cpu_percent},
                )
            )

        if memory_total_bytes and memory_total_bytes > 0:
            ratio = memory_used_bytes / memory_total_bytes
            if ratio >= 0.85:
                fingerprint = _compute_fingerprint(
                    AlertSource.METRICS, component, "memory_saturation_85"
                )
                alerts.append(
                    Alert(
                        alert_id=_generate_alert_id(),
                        source=AlertSource.METRICS,
                        timestamp=now,
                        component=component,
                        message=f"Host physical memory saturation warning: {ratio * 100:.1f}% >= 85.0% threshold",
                        severity=Severity.HIGH if ratio >= 0.95 else Severity.MEDIUM,
                        fingerprint=fingerprint,
                        recurrence_count=1,
                        raw_data={
                            "memory_used_bytes": memory_used_bytes,
                            "memory_total_bytes": memory_total_bytes,
                            "ratio": ratio,
                        },
                    )
                )

        return alerts
