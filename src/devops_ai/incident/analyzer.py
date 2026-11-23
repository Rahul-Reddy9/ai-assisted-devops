# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Tier 1 offline deterministic root cause analysis engine."""

import datetime
import time
from typing import List, Optional

from devops_ai.core.config import get_settings
from devops_ai.core.logging import get_logger
from devops_ai.incident.classifier import IncidentClassifier
from devops_ai.incident.models import (
    Category,
    Incident,
    IncidentStatus,
    RootCauseHypothesis,
    TimelineEntry,
)
from devops_ai.incident.patterns import (
    FailurePatternCatalog,
    PatternMatchResult,
    calculate_evidence_confidence,
)
from devops_ai.observability.metrics import record_rca_execution

logger = get_logger("incident.analyzer")


class IncidentAnalyzer:
    """Tier 1 deterministic root cause analyzer operating 100% offline without LLM dependencies."""

    def __init__(self, confidence_threshold: Optional[float] = None) -> None:
        settings = get_settings()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.rca_confidence_threshold
        )

    def analyze(self, incident: Incident) -> Incident:
        """Perform Tier 1 deterministic root cause analysis on an incident."""
        start_time = time.monotonic()
        now = datetime.datetime.now(datetime.timezone.utc)

        logger.info(f"Initiating Tier 1 deterministic RCA for incident '{incident.incident_id}'")

        # 1. Match Failure Patterns against Observable Evidence
        pattern_matches = FailurePatternCatalog.match_patterns(
            alerts=incident.alerts,
            evidence=incident.evidence,
        )

        # 2. Convert Matched Patterns to Hypotheses
        hypotheses: List[RootCauseHypothesis] = []
        valid_catalog_keys = set(incident.evidence.evidence_catalog.keys())
        for a in incident.alerts:
            valid_catalog_keys.add(f"EVID-ALERT-{a.alert_id}")

        for pm in pattern_matches:
            # Validate evidence IDs against actual catalog
            validated_refs = [ref for ref in pm.evidence_refs if ref in valid_catalog_keys]

            desc = pm.description
            if pm.missing_evidence:
                desc += f" (Uncorroborated: Missing {', '.join(pm.missing_evidence)})"

            conf_score = pm.confidence_breakdown.final_score if pm.confidence_breakdown else 0.50

            hypotheses.append(
                RootCauseHypothesis(
                    title=pm.title,
                    description=desc,
                    confidence=conf_score,
                    tier="tier_1_deterministic",
                    evidence_refs=validated_refs,
                    supporting_evidence=pm.supporting_evidence,
                    contradicting_evidence=pm.contradicting_evidence,
                    is_confirmed=pm.is_confirmed,
                    is_unsupported=len(validated_refs) == 0,
                )
            )

        # 3. Fallback if no deterministic pattern matched
        if not hypotheses:
            conf_breakdown = calculate_evidence_confidence(
                base_match=0.35,
                source_weight=0.70,
                corroboration=0.00,
                contradiction=0.00,
                explanation="No recognized failure pattern matched; low deterministic confidence",
            )
            fallback_refs = [f"EVID-ALERT-{a.alert_id}" for a in incident.alerts]
            hypotheses.append(
                RootCauseHypothesis(
                    title="Unknown Operational Anomaly",
                    description=(
                        "Telemetry does not match known failure patterns. "
                        "Signals suggest an operational anomaly without an established deterministic signature."
                    ),
                    confidence=conf_breakdown.final_score,
                    tier="tier_1_deterministic",
                    evidence_refs=fallback_refs,
                    supporting_evidence=[
                        f"Unclassified alert: {a.message}" for a in incident.alerts
                    ],
                    contradicting_evidence=[],
                    is_confirmed=False,
                    is_unsupported=False,
                )
            )

        # 4. Rank Hypotheses by Confidence Score
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        primary_hypothesis = hypotheses[0]
        alternative_hypotheses = hypotheses[1:]

        incident.root_cause = primary_hypothesis
        incident.alternative_hypotheses = alternative_hypotheses

        # 5. Classify Incident Category and Severity Deterministically
        primary_pm: Optional[PatternMatchResult] = pattern_matches[0] if pattern_matches else None
        class_result = IncidentClassifier.classify(incident, primary_pattern=primary_pm)
        incident.category = class_result.category
        incident.severity = class_result.severity
        incident.metadata["classification_reason"] = class_result.reason
        incident.metadata["blast_radius"] = class_result.blast_radius

        # Store confidence scoring breakdown
        if primary_pm and primary_pm.confidence_breakdown:
            incident.metadata["confidence_breakdown"] = primary_pm.confidence_breakdown.model_dump()
        else:
            incident.metadata["confidence_breakdown"] = {
                "base_score": 0.35,
                "source_weight": 0.70,
                "corroboration_bonus": 0.0,
                "contradiction_penalty": 0.0,
                "final_score": primary_hypothesis.confidence,
                "explanation": "Default fallback breakdown",
            }

        # 6. Evaluate Tier 2 AI Escalation Criteria
        escalate_to_tier_2 = False
        escalation_reasons: List[str] = []

        # Condition A: Evidence confidence below threshold (0.85)
        if primary_hypothesis.confidence < self.confidence_threshold:
            escalate_to_tier_2 = True
            escalation_reasons.append(
                f"Top hypothesis confidence ({primary_hypothesis.confidence:.2f}) is below threshold ({self.confidence_threshold:.2f})"
            )

        # Condition B: Category is APPLICATION or UNKNOWN
        if incident.category in (Category.APPLICATION, Category.UNKNOWN):
            escalate_to_tier_2 = True
            escalation_reasons.append(
                f"Incident category '{incident.category.value}' requires deep semantic investigation"
            )

        # Condition C: Competing hypotheses within 0.15 confidence margin
        if alternative_hypotheses:
            diff = abs(primary_hypothesis.confidence - alternative_hypotheses[0].confidence)
            if diff < 0.15:
                escalate_to_tier_2 = True
                escalation_reasons.append(
                    f"Ambiguous competing hypotheses with delta {diff:.2f} < 0.15 margin"
                )

        incident.metadata["escalate_to_tier_2"] = escalate_to_tier_2
        incident.metadata["escalation_reasons"] = escalation_reasons

        # 7. Lifecycle State Transition to ANALYZED
        incident.status = IncidentStatus.ANALYZED
        incident.updated_at = now

        duration_sec = time.monotonic() - start_time
        incident.timeline.append(
            TimelineEntry(
                timestamp=now,
                source=incident.alerts[0].source
                if incident.alerts
                else incident.timeline[0].source,
                component=incident.components_affected[0]
                if incident.components_affected
                else "system",
                event=(
                    f"Tier 1 deterministic RCA completed: {primary_hypothesis.title} "
                    f"(Confidence: {primary_hypothesis.confidence:.2f}, Confirmed: {primary_hypothesis.is_confirmed})"
                ),
                severity=incident.severity,
            )
        )

        record_rca_execution("tier_1", "success", duration_sec)
        logger.info(
            f"Tier 1 RCA completed for '{incident.incident_id}' in {duration_sec:.3f}s: "
            f"'{primary_hypothesis.title}' (Confidence: {primary_hypothesis.confidence:.2f})"
        )

        return incident
