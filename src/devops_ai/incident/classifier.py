# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Deterministic incident classification and evidence-based severity matrix."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from devops_ai.incident.models import Category, Incident, Severity
from devops_ai.incident.patterns import PatternMatchResult


class ClassificationResult(BaseModel):
    """Structured deterministic outcome of incident classification."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    severity: Severity
    reason: str
    evidence_ids: List[str] = Field(default_factory=list)
    blast_radius: str = "single_component"
    is_critical_justified: bool = False


class IncidentClassifier:
    """Deterministic classifier determining incident category and severity from observable evidence."""

    @staticmethod
    def classify(
        incident: Incident,
        primary_pattern: Optional[PatternMatchResult] = None,
    ) -> ClassificationResult:
        """Evaluate observable evidence and alerts to classify category and severity deterministically."""
        # 1. Determine Evidence IDs
        evidence_ids: List[str] = []
        if primary_pattern and primary_pattern.evidence_refs:
            evidence_ids.extend(primary_pattern.evidence_refs)
        for alert in incident.alerts:
            aid = f"EVID-ALERT-{alert.alert_id}"
            if aid not in evidence_ids:
                evidence_ids.append(aid)

        # 2. Determine Category
        category = Category.UNKNOWN
        category_reasons: List[str] = []

        if primary_pattern and primary_pattern.matched:
            category = primary_pattern.category
            category_reasons.append(
                f"Matched failure pattern '{primary_pattern.pattern_id}' categorized as {category.value}"
            )
        else:
            # Fall back to alert sources and components
            sources = {a.source.value for a in incident.alerts}
            if "PROVIDER" in sources or any("llm" in a.component.lower() for a in incident.alerts):
                category = Category.PROVIDER
                category_reasons.append("Alert sources and component indicate LLM provider failure")
            elif "DOCKER" in sources or "KUBERNETES" in sources:
                category = Category.CONTAINER
                category_reasons.append(
                    "Alert sources indicate Docker or Kubernetes container lifecycle failure"
                )
            elif "METRICS" in sources:
                category = Category.INFRASTRUCTURE
                category_reasons.append(
                    "Alert sources indicate host infrastructure or resource saturation"
                )
            elif "HEALTH_PROBE" in sources:
                category = Category.APPLICATION
                category_reasons.append(
                    "Health probe failures indicate application readiness degradation"
                )
            elif "LOGS" in sources:
                category = Category.APPLICATION
                category_reasons.append("Application log burst or error logs detected")
            else:
                category = Category.UNKNOWN
                category_reasons.append("Insufficient telemetry to establish failure category")

        # 3. Determine Blast Radius
        num_components = len(set(incident.components_affected))
        is_cascading = bool(incident.metadata.get("cascading_failure", False))

        if is_cascading or num_components >= 3:
            blast_radius = "systemic"
        elif num_components >= 2:
            blast_radius = "multi_component"
        else:
            blast_radius = "single_component"

        # 4. Determine Severity Deterministically
        severity = Severity.MEDIUM
        severity_reasons: List[str] = []
        is_critical_justified = False

        # Gather requested/initial alert severities
        alert_severities = [a.severity for a in incident.alerts]
        highest_alert_sev = (
            max(
                alert_severities,
                key=lambda s: ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s.value),
            )
            if alert_severities
            else Severity.MEDIUM
        )

        # Check conditions for CRITICAL severity:
        # Requires sufficiently strong evidence (confidence >= 0.85 or corroborated) AND multi-component or outage.
        pattern_conf = (
            primary_pattern.confidence_breakdown.final_score
            if (primary_pattern and primary_pattern.confidence_breakdown)
            else 0.50
        )

        has_corroborated_critical = highest_alert_sev == Severity.CRITICAL or (
            primary_pattern
            and primary_pattern.pattern_id
            in ("CONTAINER_OOM_KILLED", "K8S_CRASH_LOOP_BACKOFF", "DB_CONNECTION_REFUSED")
            and blast_radius in ("multi_component", "systemic")
        )

        if has_corroborated_critical:
            if blast_radius in ("multi_component", "systemic") and pattern_conf >= 0.80:
                severity = Severity.CRITICAL
                is_critical_justified = True
                severity_reasons.append(
                    f"CRITICAL justified: {blast_radius} outage with strong evidence confidence ({pattern_conf:.2f})"
                )
            elif pattern_conf >= 0.90 and any(
                a.severity == Severity.CRITICAL for a in incident.alerts
            ):
                severity = Severity.CRITICAL
                is_critical_justified = True
                severity_reasons.append(
                    "CRITICAL justified: High-confidence verified critical alert on core service"
                )
            else:
                # Downgrade unjustified CRITICAL
                severity = Severity.HIGH
                severity_reasons.append(
                    "Severity clamped to HIGH: CRITICAL was requested but lacks multi-component blast radius or sufficient corroborating evidence (confidence < 0.85)"
                )
        elif primary_pattern and primary_pattern.matched:
            # Use pattern default severity as baseline
            pattern_sev = primary_pattern.severity
            if pattern_sev == Severity.HIGH:
                severity = Severity.HIGH
                severity_reasons.append(
                    f"Assigned HIGH based on failure pattern '{primary_pattern.pattern_id}'"
                )
            elif pattern_sev == Severity.MEDIUM:
                if blast_radius == "multi_component":
                    severity = Severity.HIGH
                    severity_reasons.append(
                        "Escalated to HIGH: Multiple components affected by resource saturation"
                    )
                else:
                    severity = Severity.MEDIUM
                    severity_reasons.append(
                        f"Assigned MEDIUM based on failure pattern '{primary_pattern.pattern_id}'"
                    )
            else:
                severity = pattern_sev
                severity_reasons.append(
                    f"Assigned {pattern_sev.value} based on pattern '{primary_pattern.pattern_id}'"
                )
        else:
            # No pattern matched: use highest alert severity up to HIGH (never unverified CRITICAL)
            if highest_alert_sev == Severity.CRITICAL:
                severity = Severity.HIGH
                severity_reasons.append(
                    "Severity clamped to HIGH: Unverified alert claimed CRITICAL without matching known failure pattern"
                )
            elif highest_alert_sev in (Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO):
                severity = highest_alert_sev
                severity_reasons.append(
                    f"Preserved {highest_alert_sev.value} from ingested alert telemetry"
                )
            else:
                severity = Severity.MEDIUM
                severity_reasons.append("Defaulted to MEDIUM due to ambiguous evidence")

        full_reason = (
            f"Category: {'; '.join(category_reasons)}. "
            f"Severity: {'; '.join(severity_reasons)}. "
            f"Blast radius: {blast_radius} ({num_components} component(s) affected)."
        )

        return ClassificationResult(
            category=category,
            severity=severity,
            reason=full_reason,
            evidence_ids=evidence_ids,
            blast_radius=blast_radius,
            is_critical_justified=is_critical_justified,
        )
