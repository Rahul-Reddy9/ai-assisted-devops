# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Incident detector managing alert ingestion, fingerprint deduplication, and reopening."""

import datetime
from typing import Dict, List, Optional
import uuid

from devops_ai.core.config import Settings, get_settings
from devops_ai.core.logging import get_logger
from devops_ai.incident.models import (
    Alert,
    Category,
    CorrelatedEvidence,
    Incident,
    IncidentStatus,
    TimelineEntry,
)
from devops_ai.incident.repository import IncidentRepository
from devops_ai.observability.metrics import (
    record_active_incidents,
    record_incident_detected,
)

logger = get_logger("incident.detector")


class IncidentDetector:
    """Manages alert ingestion, deduplication cooldowns, and initial incident lifecycle state."""

    def __init__(
        self,
        repository: Optional[IncidentRepository] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or IncidentRepository()
        self.cooldown_seconds = self.settings.incident_dedup_cooldown_seconds
        self.reopen_window_seconds = self.settings.incident_reopen_window_seconds

    def _update_active_metrics(self) -> None:
        """Update active incidents gauge from repository state."""
        try:
            active_incidents = self.repository.list_all(active_only=True, limit=1000)
            counts: Dict[str, int] = {}
            for inc in active_incidents:
                sev = inc.severity.value
                counts[sev] = counts.get(sev, 0) + 1
            record_active_incidents(counts)
        except Exception as e:
            logger.debug(f"Failed to update active incidents metrics: {e}")

    def ingest_alert(self, alert: Alert) -> Incident:
        """Process an incoming alert, applying deduplication and reopening rules."""
        now = datetime.datetime.now(datetime.timezone.utc)

        # 1. Check existing active incidents for duplicate alerts within cooldown
        active_incidents = self.repository.list_all(limit=200)

        for existing in active_incidents:
            # Check for fingerprint match in existing alert records
            has_matching_fingerprint = any(
                a.fingerprint == alert.fingerprint for a in existing.alerts
            )

            if has_matching_fingerprint:
                # Case A: Active incident within cooldown window -> Deduplicate & increment recurrence
                if existing.status in (
                    IncidentStatus.DETECTED,
                    IncidentStatus.CORRELATED,
                    IncidentStatus.INVESTIGATING,
                    IncidentStatus.ANALYZED,
                ):
                    time_since_update = (now - existing.updated_at).total_seconds()
                    if time_since_update <= self.cooldown_seconds:
                        logger.info(
                            f"Suppressing duplicate alert '{alert.alert_id}' for incident "
                            f"'{existing.incident_id}' (cooldown active: {time_since_update:.1f}s <= {self.cooldown_seconds}s)."
                        )
                        # Increment recurrence on matching alert inside incident
                        for a in existing.alerts:
                            if a.fingerprint == alert.fingerprint:
                                a.recurrence_count += 1

                        existing.updated_at = now
                        existing.timeline.append(
                            TimelineEntry(
                                timestamp=now,
                                source=alert.source,
                                component=alert.component,
                                event=f"Recurring alert: {alert.message} (recurrence: {existing.alerts[0].recurrence_count})",
                                severity=alert.severity,
                            )
                        )
                        return self.repository.save(existing)

                # Case B: Resolved incident within reopen window -> Reopen incident
                elif existing.status == IncidentStatus.RESOLVED:
                    resolved_time = existing.resolved_at or existing.updated_at
                    time_since_resolve = (now - resolved_time).total_seconds()
                    if time_since_resolve <= self.reopen_window_seconds:
                        logger.warning(
                            f"Reopening resolved incident '{existing.incident_id}' due to recurring "
                            f"alert '{alert.alert_id}' within reopen window ({time_since_resolve:.1f}s <= {self.reopen_window_seconds}s)."
                        )
                        existing.status = IncidentStatus.DETECTED
                        existing.resolved_at = None
                        existing.updated_at = now
                        existing.timeline.append(
                            TimelineEntry(
                                timestamp=now,
                                source=alert.source,
                                component=alert.component,
                                event=f"Reopened incident: recurring alert '{alert.message}'",
                                severity=alert.severity,
                            )
                        )
                        saved = self.repository.save(existing)
                        self._update_active_metrics()
                        return saved

        # Case C: Create a new incident
        ts_str = now.strftime("%Y%m%d")
        rand_id = uuid.uuid4().hex[:6].upper()
        incident_id = f"INC-{ts_str}-{rand_id}"

        evidence_key = f"EVID-ALERT-{alert.alert_id}"
        evidence = CorrelatedEvidence(
            evidence_catalog={evidence_key: f"[{alert.source.value}] {alert.message}"},
            log_snippets=[alert.message] if alert.source.value == "LOGS" else [],
            container_states=[alert.raw_data] if alert.source.value == "DOCKER" else [],
            k8s_events=[alert.raw_data] if alert.source.value == "KUBERNETES" else [],
            probe_failures=[alert.raw_data] if alert.source.value == "HEALTH_PROBE" else [],
            metric_anomalies={alert.component: float(alert.raw_data.get("cpu_percent", 0.0))}
            if alert.source.value == "METRICS"
            else {},
        )

        incident = Incident(
            incident_id=incident_id,
            version=1,
            title=f"Incident in {alert.component}: {alert.message[:75]}",
            category=Category.UNKNOWN,
            severity=alert.severity,
            status=IncidentStatus.DETECTED,
            created_at=now,
            updated_at=now,
            components_affected=[alert.component],
            alerts=[alert],
            timeline=[
                TimelineEntry(
                    timestamp=now,
                    source=alert.source,
                    component=alert.component,
                    event=alert.message,
                    severity=alert.severity,
                )
            ],
            evidence=evidence,
            metadata={"initial_fingerprint": alert.fingerprint},
        )

        saved = self.repository.save(incident)
        record_incident_detected(severity=alert.severity.value, category="UNKNOWN")
        self._update_active_metrics()
        logger.info(
            f"Detected new incident '{incident.incident_id}' (severity: {alert.severity.value})."
        )
        return saved

    def ingest_alerts(self, alerts: List[Alert]) -> List[Incident]:
        """Ingest multiple alerts in sequence."""
        return [self.ingest_alert(alert) for alert in alerts]
