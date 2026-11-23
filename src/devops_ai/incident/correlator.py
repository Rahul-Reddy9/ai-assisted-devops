# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Multi-source incident correlation engine with sliding temporal window and dependency tracking."""

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple, Union
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

# Common directed dependency relationships: (upstream_dependency, downstream_dependent)
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

# Allowed qualifier suffixes for service instance matching
_VALID_QUALIFIERS = {
    "primary",
    "replica",
    "slave",
    "master",
    "leader",
    "follower",
    "pod",
    "svc",
    "service",
    "worker",
    "app",
    "server",
    "db",
    "node",
    "ingress",
    "egress",
    "backend",
    "frontend",
    "prod",
    "staging",
    "dev",
}


def normalize_timestamp(dt: Union[datetime.datetime, str, Any]) -> datetime.datetime:
    """Centralized timestamp normalization to timezone-aware UTC.

    Policy:
    - If dt is timezone-naive, it is explicitly assumed to represent UTC.
    - If dt has another timezone offset, it is converted to UTC.
    - If dt is an ISO 8601 string, it is parsed and normalized to UTC.
    - If dt is None or invalid, ValueError is raised to avoid silently inventing timestamps.
    """
    if dt is None:
        raise ValueError("Timestamp cannot be None")

    if isinstance(dt, str):
        try:
            dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timestamp string '{dt}': {e}") from e

    if not isinstance(dt, datetime.datetime):
        raise ValueError(f"Expected datetime object or ISO string, got {type(dt).__name__}")

    if dt.tzinfo is None:
        # Documented policy: naive datetimes are interpreted as UTC
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _normalize_component_name(name: str) -> str:
    """Normalize component or service name for comparison."""
    return name.strip().lower()


def _tokenize_component(name: str) -> List[str]:
    """Tokenize component name on delimiter boundaries (-, _, ., :, /)."""
    normalized = _normalize_component_name(name)
    return [t for t in re.split(r"[-_./:]+", normalized) if t]


def _is_token_prefix_match(tokens_a: List[str], tokens_b: List[str]) -> bool:
    """Check if one token sequence is a prefix of the other with valid qualifier suffixes.

    e.g., ['web', 'app'] and ['web', 'app', '1'] -> True
          ['api'] and ['api', 'v2'] -> True
          ['postgres'] and ['postgres', 'primary'] -> True
    Rejects:
          ['api'] and ['capital', 'mgmt'] -> False
          ['app'] and ['wrapper'] -> False
          ['auth'] and ['author', 'service'] -> False
    """
    if not tokens_a or not tokens_b:
        return False

    short_tokens, long_tokens = (
        (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    )

    # Must match the exact prefix tokens
    if long_tokens[: len(short_tokens)] != short_tokens:
        return False

    # The remaining tokens in long_tokens must be qualifier/instance identifiers
    remaining = long_tokens[len(short_tokens) :]
    for rem in remaining:
        is_num = rem.isdigit()
        is_ver = bool(re.match(r"^v\d+$", rem))
        is_hash = bool(re.match(r"^[0-9a-f]{5,}$", rem))
        if not (is_num or is_ver or is_hash or rem in _VALID_QUALIFIERS):
            return False

    return True


def _is_upstream_of(upstream_comp: str, downstream_comp: str) -> bool:
    """Check if upstream_comp is a direct dependency of downstream_comp using token matching."""
    tokens_up = set(_tokenize_component(upstream_comp))
    tokens_down = set(_tokenize_component(downstream_comp))
    for up, down in DEFAULT_DEPENDENCY_GRAPH:
        if up in tokens_up and down in tokens_down:
            return True
    return False


def _has_dependency(comp_a: str, comp_b: str) -> bool:
    """Check if two components have an upstream/downstream relationship in either direction."""
    return _is_upstream_of(comp_a, comp_b) or _is_upstream_of(comp_b, comp_a)


def _components_are_related(comp_a: str, comp_b: str) -> bool:
    """Determine whether two components are identical, shared service instance, or have a direct dependency."""
    a = _normalize_component_name(comp_a)
    b = _normalize_component_name(comp_b)

    if a == b:
        return True

    tokens_a = _tokenize_component(a)
    tokens_b = _tokenize_component(b)

    # Delimiter-aware prefix/instance matching
    if _is_token_prefix_match(tokens_a, tokens_b):
        return True

    # Dependency graph match (upstream <-> downstream)
    if _has_dependency(comp_a, comp_b):
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
        1. Both occur within the sliding temporal window (evaluated in UTC).
        2. AND have an identity, delimiter-aware service instance, dependency, or shared container link.
        Two completely unrelated components occurring at the same time are NOT correlated.
        """
        # 1. Temporal Proximity check with normalized UTC timestamps
        t_a = normalize_timestamp(alert_a.timestamp)
        t_b = normalize_timestamp(alert_b.timestamp)
        dt = abs((t_a - t_b).total_seconds())
        if dt > self.window_seconds:
            return False

        # 2. Component or dependency relationship (delimiter-aware)
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

        # 4. Shared host/node name with metric/container context
        host_a = alert_a.raw_data.get("host") or alert_a.raw_data.get("node")
        host_b = alert_b.raw_data.get("host") or alert_b.raw_data.get("node")
        if host_a and host_b and str(host_a) == str(host_b) and str(host_a).strip():
            if alert_a.source == AlertSource.METRICS or alert_b.source == AlertSource.METRICS:
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
        alert_ts = normalize_timestamp(alert.timestamp)

        # Search for a matching active incident within the correlation window
        for inc in active_incidents:
            if inc.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED):
                continue

            inc_ts = normalize_timestamp(inc.updated_at)
            time_diff = abs((alert_ts - inc_ts).total_seconds())
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

        # Sort chronologically, handling timezone-aware and naive datetimes safely
        sorted_alerts = sorted(alerts, key=lambda a: normalize_timestamp(a.timestamp))
        incidents: List[Incident] = []

        for alert in sorted_alerts:
            alert_ts = normalize_timestamp(alert.timestamp)
            matched_inc: Optional[Incident] = None
            for inc in incidents:
                inc_ts = normalize_timestamp(inc.updated_at)
                time_diff = abs((alert_ts - inc_ts).total_seconds())
                if time_diff <= self.window_seconds:
                    if any(
                        self.are_alerts_correlatable(alert, existing) for existing in inc.alerts
                    ):
                        matched_inc = inc
                        break

            if matched_inc:
                self._merge_alert_into_incident(matched_inc, alert)
            else:
                new_inc = self._create_incident_from_alert(alert, alert_ts)
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
        norm_ts = normalize_timestamp(alert.timestamp)
        incident.timeline.append(
            TimelineEntry(
                timestamp=norm_ts,
                source=alert.source,
                component=alert.component,
                event=f"Correlated alert: {alert.message}",
                severity=alert.severity,
            )
        )

        # Detect cascading failures (participating subgraph only)
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

        inc_ts = normalize_timestamp(incident.updated_at)
        incident.updated_at = max(inc_ts, norm_ts)

    def _create_incident_from_alert(self, alert: Alert, now: datetime.datetime) -> Incident:
        """Create a new incident initialized from a single alert."""
        norm_ts = normalize_timestamp(now)
        alert_ts = normalize_timestamp(alert.timestamp)
        ts_str = norm_ts.strftime("%Y%m%d")
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
            created_at=alert_ts,
            updated_at=alert_ts,
            components_affected=[alert.component],
            alerts=[alert],
            timeline=[
                TimelineEntry(
                    timestamp=alert_ts,
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
        """Detect whether alerts form a cascading failure chain based on timestamps and dependencies.

        Returns ONLY the participating dependency subgraph in causal timestamp order.
        Unrelated earlier alerts are never included as cascade roots.
        """
        if len(components) < 2:
            return []

        # Find chronologically ordered components using normalized UTC timestamps
        comp_first_seen: Dict[str, datetime.datetime] = {}
        for a in sorted(alerts, key=lambda x: normalize_timestamp(x.timestamp)):
            if a.component not in comp_first_seen:
                comp_first_seen[a.component] = normalize_timestamp(a.timestamp)

        ordered_comps = sorted(comp_first_seen.keys(), key=lambda c: comp_first_seen[c])

        # Find the longest directed dependency path among ordered components
        longest_chain: List[str] = []
        for i, start_comp in enumerate(ordered_comps):
            current_chain = [start_comp]
            for next_comp in ordered_comps[i + 1 :]:
                if _is_upstream_of(current_chain[-1], next_comp):
                    current_chain.append(next_comp)
            if len(current_chain) > len(longest_chain):
                longest_chain = current_chain

        return longest_chain if len(longest_chain) >= 2 else []
