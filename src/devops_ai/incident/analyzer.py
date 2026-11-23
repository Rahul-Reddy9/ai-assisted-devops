# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Two-tier root cause analysis engine combining deterministic Tier 1 and AI-assisted Tier 2."""

import datetime
import time
from typing import List, Optional

from devops_ai.core.config import get_settings
from devops_ai.core.logging import get_logger
from devops_ai.incident.ai_engine import Tier2AIEngine
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
from devops_ai.incident.remediation import RemediationAdvisor
from devops_ai.observability.metrics import record_rca_execution

logger = get_logger("incident.analyzer")


class IncidentAnalyzer:
    """Two-tier incident root cause analyzer with deterministic Tier 1 and AI-assisted Tier 2."""

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        auto_tier2: Optional[bool] = None,
        ai_engine: Optional[Tier2AIEngine] = None,
        remediation_advisor: Optional[RemediationAdvisor] = None,
    ) -> None:
        settings = get_settings()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.rca_confidence_threshold
        )
        self.auto_tier2 = auto_tier2 if auto_tier2 is not None else settings.rca_auto_tier2
        self.ai_engine = ai_engine or Tier2AIEngine()
        self.remediation_advisor = remediation_advisor or RemediationAdvisor()

    def analyze(
        self,
        incident: Incident,
        force_tier2: bool = False,
        skip_tier2: bool = False,
    ) -> Incident:
        """Perform two-tier root cause analysis: deterministic Tier 1 followed by conditional Tier 2."""
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
            is_unsupported = len(validated_refs) == 0

            desc = pm.description
            if pm.missing_evidence:
                desc += f" (Uncorroborated: Missing {', '.join(pm.missing_evidence)})"

            conf_score = pm.confidence_breakdown.final_score if pm.confidence_breakdown else 0.50
            if is_unsupported:
                # Cap unsupported hypothesis confidence at 0.30 before ranking and escalation
                conf_score = min(conf_score, 0.30)
                desc += " [UNSUPPORTED: No valid evidence citations found in catalog; confidence capped at 0.30]"

            hypotheses.append(
                RootCauseHypothesis(
                    title=pm.title,
                    description=desc,
                    confidence=conf_score,
                    tier="tier_1_deterministic",
                    evidence_refs=validated_refs,
                    supporting_evidence=pm.supporting_evidence,
                    contradicting_evidence=pm.contradicting_evidence,
                    is_confirmed=pm.is_confirmed and not is_unsupported,
                    is_unsupported=is_unsupported,
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

        # Store Tier 1 summary and confidence scoring breakdown
        incident.metadata["tier_1_result"] = primary_hypothesis.model_dump()
        if primary_pm and primary_pm.confidence_breakdown:
            cb_dict = primary_pm.confidence_breakdown.model_dump()
            if primary_hypothesis.is_unsupported:
                cb_dict["uncapped_score"] = cb_dict["final_score"]
                cb_dict["final_score"] = primary_hypothesis.confidence
                cb_dict["is_unsupported"] = True
                cb_dict["capped_reason"] = (
                    "Confidence capped at 0.30 because no verified evidence citations exist in catalog"
                )
            incident.metadata["confidence_breakdown"] = cb_dict
        else:
            incident.metadata["confidence_breakdown"] = {
                "base_score": 0.35,
                "source_weight": 0.70,
                "corroboration_bonus": 0.0,
                "contradiction_penalty": 0.0,
                "final_score": primary_hypothesis.confidence,
                "is_unsupported": primary_hypothesis.is_unsupported,
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

        # Condition D: Top hypothesis is unsupported by verified evidence references
        if primary_hypothesis.is_unsupported:
            escalate_to_tier_2 = True
            escalation_reasons.append(
                "Top hypothesis is unsupported by verified evidence references in catalog"
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

        # 8. Conditional Tier 2 AI Investigation Orchestration
        should_escalate = (
            incident.metadata.get("escalate_to_tier_2", False) or force_tier2
        ) and not skip_tier2

        # Avoid unnecessary repeated AI requests if already attempted and not forced
        if incident.metadata.get("tier_2_attempted") is True and not force_tier2:
            should_escalate = False

        if should_escalate and (self.auto_tier2 or force_tier2):
            t2_start = time.monotonic()
            ai_hypothesis = self.ai_engine.investigate(incident)
            t2_duration = time.monotonic() - t2_start

            if ai_hypothesis:
                # Tier 2 AI investigation succeeded
                t1_primary = incident.root_cause

                can_override = False
                if t1_primary is None:
                    can_override = True
                elif t1_primary.is_confirmed:
                    # Confirmed deterministic Tier 1 root cause requires confirmed Tier 2 with strictly higher confidence
                    if (
                        ai_hypothesis.is_confirmed
                        and ai_hypothesis.confidence > t1_primary.confidence
                    ):
                        can_override = True
                else:
                    # Unconfirmed Tier 1 can be overridden if AI confidence >= Tier 1 confidence or if Tier 1 was fallback
                    if (
                        ai_hypothesis.confidence >= t1_primary.confidence
                        or t1_primary.title == "Unknown Operational Anomaly"
                    ):
                        can_override = True

                conf_val = (
                    float(ai_hypothesis.confidence)
                    if isinstance(ai_hypothesis.confidence, (int, float))
                    else 0.80
                )
                t2_now = datetime.datetime.now(datetime.timezone.utc)

                if can_override:
                    incident.root_cause = ai_hypothesis
                    if t1_primary and t1_primary not in incident.alternative_hypotheses:
                        incident.alternative_hypotheses.insert(0, t1_primary)
                    incident.metadata["root_cause_precedence"] = "tier_2_override"
                    timeline_event = (
                        f"Tier 2 AI RCA completed: {ai_hypothesis.title} "
                        f"(Confidence: {conf_val:.2f}, Confirmed: {ai_hypothesis.is_confirmed})"
                    )
                else:
                    # Preserve authoritative Tier 1 finding as primary; store AI hypothesis in alternatives
                    if ai_hypothesis not in incident.alternative_hypotheses:
                        incident.alternative_hypotheses.insert(0, ai_hypothesis)
                    incident.metadata["root_cause_precedence"] = "tier_1_preserved"
                    incident.metadata["tier_2_override_rejected_reason"] = (
                        f"Tier 1 finding '{t1_primary.title}' (confirmed: {t1_primary.is_confirmed}, "
                        f"conf: {t1_primary.confidence:.2f}) preserved over Tier 2 hypothesis '{ai_hypothesis.title}' "
                        f"(confirmed: {ai_hypothesis.is_confirmed}, conf: {conf_val:.2f})"
                    )
                    timeline_event = (
                        f"Tier 2 AI RCA completed: preserved confirmed Tier 1 RCA '{t1_primary.title}'; "
                        f"stored AI hypothesis in alternatives"
                    )

                incident.timeline.append(
                    TimelineEntry(
                        timestamp=t2_now,
                        source=incident.alerts[0].source
                        if incident.alerts
                        else incident.timeline[0].source,
                        component=incident.components_affected[0]
                        if incident.components_affected
                        else "system",
                        event=timeline_event,
                        severity=incident.severity,
                    )
                )
                record_rca_execution("tier_2", "success", t2_duration)
                logger.info(
                    f"Tier 2 AI RCA completed for '{incident.incident_id}': '{ai_hypothesis.title}' "
                    f"(Confidence: {conf_val:.2f}, override: {can_override})"
                )
            else:
                # Tier 2 failed or was ungrounded; preserve Tier 1 results
                incident.metadata["tier_2_fallback_used"] = True
                t2_now = datetime.datetime.now(datetime.timezone.utc)
                incident.timeline.append(
                    TimelineEntry(
                        timestamp=t2_now,
                        source=incident.alerts[0].source
                        if incident.alerts
                        else incident.timeline[0].source,
                        component=incident.components_affected[0]
                        if incident.components_affected
                        else "system",
                        event="Tier 2 AI RCA failed or ungrounded; preserved Tier 1 deterministic RCA",
                        severity=incident.severity,
                    )
                )
                record_rca_execution("tier_2", "fallback", t2_duration)
                logger.warning(
                    f"Tier 2 AI RCA failed for '{incident.incident_id}'; preserved Tier 1 deterministic RCA: "
                    f"'{primary_hypothesis.title}'"
                )
        else:
            # Tier 2 skipped
            incident.metadata["tier_2_attempted"] = False
            incident.metadata["tier_2_skipped"] = True
            if not (incident.metadata.get("escalate_to_tier_2", False) or force_tier2):
                incident.metadata["tier_2_skip_reason"] = (
                    "Deterministic Tier 1 confidence is sufficiently high; escalation criteria not triggered"
                )
            else:
                incident.metadata["tier_2_skip_reason"] = (
                    "Tier 2 escalation criteria met but auto_tier2 is disabled"
                )

        # 9. Generate Advisory Remediation Recommendations
        recommendations = self.remediation_advisor.generate_recommendations(incident)
        incident.recommendations = recommendations
        incident.metadata["remediation_generated"] = True

        return incident
