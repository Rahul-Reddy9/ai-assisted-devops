# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Multi-source incident correlation engine with sliding temporal window and dependency tracking."""

import datetime
from typing import Dict, List, Optional, Tuple
import uuid

from devops_ai.core.config import get_settings
from devops_ai.core.logging import get_logger
from devops_ai.incident.models import (
    Alert,
    AlertSource,
    Category,
    CorrelatedEvidence,
    Incident,
    IncidentStatus,
    TimelineEntry,
)

logger = get_logger("incident.correlator")

# Common dependency relationships: (dependency/upstream, dependent/downstream)
DEFAULT_DEPENDENCY_GRAPH: List[Tuple[str, str]] = [
    ("postgres", "api"),
    ("postgres", "backend"),
    ("postgres", "web"),
    ("database", "api"),
    ("database", "backend"),
    ("database", "web"),
    ("redis", "worker"),
    ("redis", "queue"),
    ("redis", "api"),
    ("backend", "frontend"),
    ("backend", "web"),
    ("api", "frontend"),
    ("api", "web"),
]


def _normalize_component_name(name: str) -> str:
    """Normalize component or service name for comparison."""
    return name.strip().lower()


def _components_are_related(comp_a: str, comp_b: str) -> bool:
    """Determine whether two components are identical, shared service, or have a direct dependency."""
    a = _normalize_component_name(comp_a)
    b = _normalize_component_name(comp_b)

    if a == b:
        return True

    # Substring / container prefix match (e.g. 'web-app-1' and 'web-app', 'pod-api-xyz' and 'api')
    if a in b or b in a:
        return True

    # Dependency graph match (upstream -> downstream or downstream -> upstream)
    for upstream, downstream in DEFAULT_DEPENDENCY_GRAPH:
        if (upstream in a and downstream in b) or (upstream in b and downstream in a):
            return True

    return False


def _safe_truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate excessively large log or evidence text safely without throwing."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2 - 20
    return text[:half] + "\n... [TRUNCATED] ...\n" + text[-half:]


class IncidentCorrelator:
    """Correlates multi-source observability alerts into unified, context-rich incidents."""

    def __init__(self, window_seconds: Optional[int] = None) -> None:
        settings = get_settings()
        self.window_seconds = (
            window_seconds
            if window_seconds is not None
            else settings.incident_correlation_window_seconds
        )

    def are_alerts_correlatable(self, alert_a: Alert, alert_b: Alert) -> bool:
        """Determine whether two alerts should be correlated into the same incident.

        Correlated if:
        1. Both occur within the sliding temporal window.
        2. AND have an identity, dependency, or shared evidence link.
        Two completely unrelated components occurring at the same time are NOT correlated.
        """
        # 1. Temporal Proximity check
        dt = abs((alert_a.timestamp - alert_b.timestamp).total_seconds())
        if dt > self.window_seconds:
            return False

        # 2. Component or dependency relationship
        if _components_are_related(alert_a.component, alert_b.component):
            return True

        # 3. Shared container ID or pod name
        container_a = alert_a.raw_data.get("container_id") or alert_a.raw_data.get("container")
        container_b = alert_b.raw_data.get("container_id") or alert_b.raw_data.get("container")
        if container_a and container_b and str(container_a) == str(container_b):
            return True

        pod_a = alert_a.raw_data.get("pod_name") or alert_a.raw_data.get("pod")
        pod_b = alert_b.raw_data.get("pod_name") or alert_b.raw_data.get("pod")
        if pod_a and pod_b and str(pod_a) == str(pod_b):
            return True

        # 4. Shared specific error pattern or signature (e.g. both reporting OOM or same host exhaustion)
        msg_a = alert_a.message.lower()
        msg_b = alert_b.message.lower()
        shared_keywords = ("oom", "137", "connection refused", "econnrefused", "rate limit")
        for kw in shared_keywords:
            if kw in msg_a and kw in msg_b:
                return True

        return False

    def correlate_alert(
        self,
        alert: Alert,
        active_incidents: List[Incident],
    ) -> Incident:
        """Correlate a single alert into an existing active incident, or create a new incident."""
        now = datetime.datetime.now(datetime.timezone.utc)
        target_incident: Optional[Incident] = None

        # Search for a matching active incident within the correlation window
        for inc in active_incidents:
            if inc.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED):
                continue

            time_diff = abs((alert.timestamp - inc.updated_at).total_seconds())
            if time_diff > self.window_seconds:
                continue

            # Check if alert matches any existing alert in this incident
            is_match = any(
                self.are_alerts_correlatable(alert, existing_alert) for existing_alert in inc.alerts
            )
            if is_match:
                target_incident = inc
                break

        if target_incident:
            self._merge_alert_into_incident(target_incident, alert)
            return target_incident

        # Create new incident if no correlation match found
        return self._create_incident_from_alert(alert, now)

    def correlate_batch(self, alerts: List[Alert]) -> List[Incident]:
        """Correlate a batch of alerts chronologically into grouped incidents."""
        if not alerts:
            return []

        # Sort chronologically, handling timezone-aware datetimes
        sorted_alerts = sorted(alerts, key=lambda a: a.timestamp)
        incidents: List[Incident] = []

        for alert in sorted_alerts:
            matched_inc: Optional[Incident] = None
            for inc in incidents:
                time_diff = abs((alert.timestamp - inc.updated_at).total_seconds())
                if time_diff <= self.window_seconds:
                    if any(
                        self.are_alerts_correlatable(alert, existing) for existing in inc.alerts
                    ):
                        matched_inc = inc
                        break

            if matched_inc:
                self._merge_alert_into_incident(matched_inc, alert)
            else:
                new_inc = self._create_incident_from_alert(alert, alert.timestamp)
                incidents.append(new_inc)

        return incidents

    def _merge_alert_into_incident(self, incident: Incident, alert: Alert) -> None:
        """Merge an incoming alert into an existing incident, updating evidence and timeline."""
        # Check for duplicate alert
        is_duplicate = False
        for existing in incident.alerts:
            if existing.fingerprint == alert.fingerprint:
                existing.recurrence_count += alert.recurrence_count
                is_duplicate = True
                break

        if not is_duplicate:
            incident.alerts.append(alert)

        # Update affected components
        if alert.component not in incident.components_affected:
            incident.components_affected.append(alert.component)

        # Update evidence bundle without duplicating identical catalog entries
        self._add_alert_to_evidence(incident.evidence, alert)

        # Append timeline entry
        incident.timeline.append(
            TimelineEntry(
                timestamp=alert.timestamp,
                source=alert.source,
                component=alert.component,
                event=f"Correlated alert: {alert.message}",
                severity=alert.severity,
            )
        )

        # Detect cascading failures
        cascade_chain = self._detect_cascade(incident.components_affected, incident.alerts)
        if cascade_chain:
            incident.metadata["cascading_failure"] = True
            incident.metadata["cascade_chain"] = cascade_chain

        # Escalate incident status to CORRELATED if currently in DETECTED
        if incident.status == IncidentStatus.DETECTED:
            incident.status = IncidentStatus.CORRELATED

        # Escalate severity if incoming alert is higher
        sev_rank = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if sev_rank.index(alert.severity.value) > sev_rank.index(incident.severity.value):
            incident.severity = alert.severity

        incident.updated_at = max(incident.updated_at, alert.timestamp)

    def _create_incident_from_alert(self, alert: Alert, now: datetime.datetime) -> Incident:
        """Create a new incident initialized from a single alert."""
        ts_str = now.strftime("%Y%m%d")
        rand_id = uuid.uuid4().hex[:6].upper()
        incident_id = f"INC-{ts_str}-{rand_id}"

        evidence = CorrelatedEvidence()
        self._add_alert_to_evidence(evidence, alert)

        return Incident(
            incident_id=incident_id,
            version=1,
            title=f"Incident in {alert.component}: {alert.message[:75]}",
            category=Category.UNKNOWN,
            severity=alert.severity,
            status=IncidentStatus.DETECTED,
            created_at=alert.timestamp,
            updated_at=alert.timestamp,
            components_affected=[alert.component],
            alerts=[alert],
            timeline=[
                TimelineEntry(
                    timestamp=alert.timestamp,
                    source=alert.source,
                    component=alert.component,
                    event=f"Initial alert: {alert.message}",
                    severity=alert.severity,
                )
            ],
            evidence=evidence,
        )

    def _add_alert_to_evidence(self, evidence: CorrelatedEvidence, alert: Alert) -> None:
        """Ingest alert payload into evidence catalog and specialized sub-collections."""
        source_prefix = {
            AlertSource.LOGS: "LOG",
            AlertSource.METRICS: "METRIC",
            AlertSource.HEALTH_PROBE: "PROBE",
            AlertSource.DOCKER: "DOCKER",
            AlertSource.KUBERNETES: "K8S",
            AlertSource.MANUAL: "MANUAL",
        }.get(alert.source, "GEN")

        # Deterministic evidence ID based on alert source and id
        evid_id = f"EVID-{source_prefix}-{alert.alert_id[-8:]}"

        # Prevent duplicate insertion of identical evidence content
        if evid_id not in evidence.evidence_catalog:
            truncated_msg = _safe_truncate_text(
                f"[{alert.source.value}] ({alert.component}) {alert.message}"
            )
            evidence.evidence_catalog[evid_id] = truncated_msg

        # Specialized bucket mapping
        if alert.source == AlertSource.LOGS:
            if alert.message not in evidence.log_snippets and len(evidence.log_snippets) < 100:
                evidence.log_snippets.append(_safe_truncate_text(alert.message, 1000))
        elif alert.source == AlertSource.DOCKER:
            if alert.raw_data and alert.raw_data not in evidence.container_states:
                evidence.container_states.append(alert.raw_data)
        elif alert.source == AlertSource.KUBERNETES:
            if alert.raw_data and alert.raw_data not in evidence.k8s_events:
                evidence.k8s_events.append(alert.raw_data)
        elif alert.source == AlertSource.HEALTH_PROBE:
            if alert.raw_data and alert.raw_data not in evidence.probe_failures:
                evidence.probe_failures.append(alert.raw_data)
        elif alert.source == AlertSource.METRICS:
            val = float(
                alert.raw_data.get("cpu_percent") or alert.raw_data.get("memory_percent") or 0.0
            )
            evidence.metric_anomalies[alert.component] = val

    def _detect_cascade(self, components: List[str], alerts: List[Alert]) -> List[str]:
        """Detect whether alerts form a cascading failure chain based on timestamps and dependencies."""
        if len(components) < 2:
            return []

        # Find chronologically ordered components
        comp_first_seen: Dict[str, datetime.datetime] = {}
        for a in sorted(alerts, key=lambda x: x.timestamp):
            if a.component not in comp_first_seen:
                comp_first_seen[a.component] = a.timestamp

        ordered_comps = sorted(comp_first_seen.keys(), key=lambda c: comp_first_seen[c])

        # Verify dependency order
        for i in range(len(ordered_comps) - 1):
            c1 = _normalize_component_name(ordered_comps[i])
            c2 = _normalize_component_name(ordered_comps[i + 1])
            for upstream, downstream in DEFAULT_DEPENDENCY_GRAPH:
                if upstream in c1 and downstream in c2:
                    return ordered_comps

        return []
