# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Deterministic failure pattern catalog and signature matching for Tier 1 RCA."""

import re
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from devops_ai.incident.models import Alert, Category, CorrelatedEvidence, Severity


class ConfidenceBreakdown(BaseModel):
    """Detailed mathematical breakdown of the evidence confidence score.

    Formula:
        C_det = clamp((S_base * W_source) + S_corrob - P_contradict, 0.0, 1.0)
    Note:
        This is an evidence confidence score, NOT a calibrated probability.
    """

    model_config = ConfigDict(extra="forbid")

    base_score: float = Field(ge=0.0, le=1.0)
    source_weight: float = Field(ge=0.0, le=1.0)
    corroboration_bonus: float = Field(ge=0.0, le=0.25)
    contradiction_penalty: float = Field(ge=0.0, le=0.50)
    final_score: float = Field(ge=0.0, le=1.0)
    explanation: str


def calculate_evidence_confidence(
    base_match: float,
    source_weight: float,
    corroboration: float = 0.0,
    contradiction: float = 0.0,
    explanation: str = "",
) -> ConfidenceBreakdown:
    """Calculate the deterministic evidence confidence score with bounded clamping.

    Args:
        base_match: Base evidence strength (S_base), clamped between 0.30 and 1.00.
        source_weight: Source reliability weight (W_source), clamped between 0.60 and 1.00.
        corroboration: Independent corroboration bonus (S_corrob), clamped between 0.00 and 0.25.
        contradiction: Contradiction penalty (P_contradict), clamped between 0.00 and 0.50.
        explanation: Human-readable scoring explanation.

    Returns:
        ConfidenceBreakdown containing the exact inputs and clamped final score.
    """
    s_base = max(0.30, min(1.00, float(base_match)))
    w_source = max(0.60, min(1.00, float(source_weight)))
    s_corrob = max(0.00, min(0.25, float(corroboration)))
    p_contra = max(0.00, min(0.50, float(contradiction)))

    raw_score = (s_base * w_source) + s_corrob - p_contra
    final_score = max(0.00, min(1.00, round(raw_score, 4)))

    if not explanation:
        explanation = (
            f"Evidence confidence: (S_base={s_base:.2f} * W_source={w_source:.2f}) "
            f"+ S_corrob={s_corrob:.2f} - P_contradict={p_contra:.2f} = {final_score:.4f}"
        )

    return ConfidenceBreakdown(
        base_score=s_base,
        source_weight=w_source,
        corroboration_bonus=s_corrob,
        contradiction_penalty=p_contra,
        final_score=final_score,
        explanation=explanation,
    )


class PatternMatchResult(BaseModel):
    """Result of evaluating a single failure pattern against incident evidence."""

    model_config = ConfigDict(extra="forbid")

    pattern_id: str
    matched: bool
    title: str
    description: str
    category: Category
    severity: Severity
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    evidence_refs: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    is_confirmed: bool = False
    diagnostic_explanation: str = ""


# Regular expressions for deterministic failure matching
RE_OOM_KERNEL = re.compile(
    r"(?i)(out of memory: kill process|invoked oom-killer|oom_reaper|oom-kill|killed process \d+ \([^)]+\) total-vm:\d+kB)"
)
RE_SIGKILL_EXIT = re.compile(r"(?i)(exit(?:ed)?\s*\(137\)|exit code 137|code[:\s]+137)")
RE_DB_CONN_REFUSED = re.compile(
    r"(?i)(connection refused|econnrefused|psycopg2\.operationalerror|could not connect to server|dial tcp [^:]+:\d+: connect: connection refused|connection to [^ ]+ failed)"
)
RE_CRASH_LOOP = re.compile(r"(?i)(crashloopbackoff|back-off \d+s restarting failed container)")
RE_IMAGE_PULL = re.compile(
    r"(?i)(imagepullbackoff|errimagepull|failed to pull image|image inspect error|unauthorized: authentication required)"
)
RE_UPSTREAM_TIMEOUT = re.compile(
    r"(?i)(request timeout|gateway timeout|timed out after|deadline exceeded|504 gateway time-out|upstream request timeout)"
)
RE_LLM_RATE_LIMIT = re.compile(
    r"(?i)(429 too many requests|rate limit exceeded|quota exceeded|resource_exhausted|tpm limit|rpm limit)"
)


class FailurePatternCatalog:
    """Catalog of deterministic failure patterns with strict SIGKILL/OOM distinction."""

    @staticmethod
    def match_patterns(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
    ) -> List[PatternMatchResult]:
        """Evaluate all known failure patterns against the incident alerts and evidence catalog.

        Returns matching PatternMatchResult items ranked by evidence confidence score.
        """
        results: List[PatternMatchResult] = []

        # Aggregate evidence text for regex matching
        combined_text_entries: List[tuple[str, str]] = []  # (evid_id, text)
        for eid, text in evidence.evidence_catalog.items():
            combined_text_entries.append((eid, text))

        for alert in alerts:
            combined_text_entries.append(
                (f"EVID-ALERT-{alert.alert_id}", f"{alert.message} {alert.raw_data}")
            )

        # 1. Pattern: Confirmed OOMKilled vs Ambiguous SIGKILL
        oom_res = FailurePatternCatalog._evaluate_oom_and_sigkill(
            alerts, evidence, combined_text_entries
        )
        if oom_res:
            results.append(oom_res)

        # 2. Pattern: K8s CrashLoopBackOff
        crash_res = FailurePatternCatalog._evaluate_crash_loop(
            alerts, evidence, combined_text_entries
        )
        if crash_res:
            results.append(crash_res)

        # 3. Pattern: K8s ImagePullBackOff
        img_res = FailurePatternCatalog._evaluate_image_pull(
            alerts, evidence, combined_text_entries
        )
        if img_res:
            results.append(img_res)

        # 4. Pattern: Database Connection Refused
        db_res = FailurePatternCatalog._evaluate_db_connection(
            alerts, evidence, combined_text_entries
        )
        if db_res:
            results.append(db_res)

        # 5. Pattern: CPU Throttling
        cpu_res = FailurePatternCatalog._evaluate_cpu_throttling(
            alerts, evidence, combined_text_entries
        )
        if cpu_res:
            results.append(cpu_res)

        # 6. Pattern: Memory Exhaustion (without confirmed OOM)
        mem_res = FailurePatternCatalog._evaluate_memory_exhaustion(
            alerts, evidence, combined_text_entries, oom_res
        )
        if mem_res:
            results.append(mem_res)

        # 7. Pattern: Readiness Probe Failure & Cascading Service Unavailability
        probe_res = FailurePatternCatalog._evaluate_readiness_probe(
            alerts, evidence, combined_text_entries
        )
        if probe_res:
            results.append(probe_res)

        # 8. Pattern: Upstream Request Timeout
        timeout_res = FailurePatternCatalog._evaluate_upstream_timeout(
            alerts, evidence, combined_text_entries
        )
        if timeout_res:
            results.append(timeout_res)

        # 9. Pattern: LLM Provider Rate Limiting
        rate_res = FailurePatternCatalog._evaluate_llm_rate_limiting(
            alerts, evidence, combined_text_entries
        )
        if rate_res:
            results.append(rate_res)

        # Filter to only matched patterns and sort descending by final confidence score
        matched = [r for r in results if r.matched and r.confidence_breakdown]
        matched.sort(
            key=lambda r: r.confidence_breakdown.final_score if r.confidence_breakdown else 0.0,
            reverse=True,
        )
        return matched

    @staticmethod
    def _evaluate_oom_and_sigkill(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Strictly distinguish confirmed OOMKilled from ambiguous exit code 137 (SIGKILL)."""
        has_exit_137 = False
        exit_evid_refs: List[str] = []

        # Check container states for exit code 137
        for cs in evidence.container_states:
            st = str(cs.get("status") or "")
            code = cs.get("exit_code")
            if code == 137 or RE_SIGKILL_EXIT.search(st):
                has_exit_137 = True

        for eid, text in text_entries:
            if RE_SIGKILL_EXIT.search(text):
                has_exit_137 = True
                if eid not in exit_evid_refs:
                    exit_evid_refs.append(eid)

        if not has_exit_137:
            return None

        # Check for confirmed OOM corroboration:
        # a) Docker State.OOMKilled == True
        # b) K8s termination reason == "OOMKilled"
        # c) Kernel / dmesg regex match
        confirmed_oom = False
        oom_evid_refs: List[str] = []
        oom_supporting: List[str] = []

        for cs in evidence.container_states:
            if cs.get("oom_killed") is True:
                confirmed_oom = True
                oom_supporting.append("Docker State.OOMKilled flag is true")

        for k8s in evidence.k8s_events:
            reason = str(k8s.get("reason") or "")
            msg = str(k8s.get("message") or "")
            if "oomkilled" in reason.lower() or "oomkilled" in msg.lower():
                confirmed_oom = True
                oom_supporting.append(f"Kubernetes container terminated with OOMKilled ({reason})")

        for eid, text in text_entries:
            if RE_OOM_KERNEL.search(text):
                confirmed_oom = True
                if eid not in oom_evid_refs:
                    oom_evid_refs.append(eid)
                oom_supporting.append(f"Kernel OOM-killer invocation recorded in log [{eid}]")

        # Memory metric check for corroboration
        mem_spike = any(v >= 90.0 for v in evidence.metric_anomalies.values())
        for a in alerts:
            if a.source.value == "METRICS" and a.raw_data.get("memory_percent", 0.0) >= 90.0:
                mem_spike = True

        if confirmed_oom:
            # Confirmed OOMKilled
            all_refs = list(dict.fromkeys(exit_evid_refs + oom_evid_refs))
            corrob_bonus = 0.15 if mem_spike else 0.05
            if mem_spike:
                oom_supporting.append(
                    "Corroborating host/container memory utilization > 90% recorded"
                )

            confidence = calculate_evidence_confidence(
                base_match=1.00,
                source_weight=0.95,
                corroboration=corrob_bonus,
                contradiction=0.00,
                explanation="Confirmed OOM: Verified exit code 137 with kernel or container OOMKilled status",
            )
            return PatternMatchResult(
                pattern_id="CONTAINER_OOM_KILLED",
                matched=True,
                title="Container Terminated by Linux Kernel Out-Of-Memory Killer (OOMKilled)",
                description=(
                    "The container was terminated by the Linux kernel OOM killer after exceeding "
                    "its provisioned memory limit or causing host memory exhaustion."
                ),
                category=Category.CONTAINER,
                severity=Severity.HIGH,
                confidence_breakdown=confidence,
                evidence_refs=all_refs,
                supporting_evidence=oom_supporting,
                contradicting_evidence=[],
                missing_evidence=[],
                is_confirmed=True,
                diagnostic_explanation=(
                    "Process allocated memory beyond cgroup limit. Kernel invoked oom-killer and sent SIGKILL."
                ),
            )
        else:
            # Ambiguous SIGKILL (exit code 137 alone)
            confidence = calculate_evidence_confidence(
                base_match=0.60,
                source_weight=0.85,
                corroboration=0.00,
                contradiction=0.00,
                explanation="Ambiguous SIGKILL: Exit code 137 alone without confirmed OOM corroboration",
            )
            supporting = ["Container process terminated with status code 137 (SIGKILL)"]
            if mem_spike:
                supporting.append(
                    "High memory utilization observed (>90%), but no kernel or container OOMKilled status found"
                )
            missing = [
                "Kubernetes pod termination reason OOMKilled",
                "Docker container State.OOMKilled flag",
                "Explicit kernel dmesg or cgroup Out of memory log lines",
            ]
            return PatternMatchResult(
                pattern_id="CONTAINER_SIGKILL_TERMINATION",
                matched=True,
                title="Container Forcefully Terminated with SIGKILL (Exit Code 137)",
                description=(
                    "Container exited with code 137 (SIGKILL). This indicates forced termination, which may be "
                    "caused by an OOM kill, manual docker kill, orchestrator preemption, or failed health check."
                ),
                category=Category.CONTAINER,
                severity=Severity.HIGH,
                confidence_breakdown=confidence,
                evidence_refs=exit_evid_refs,
                supporting_evidence=supporting,
                contradicting_evidence=[],
                missing_evidence=missing,
                is_confirmed=False,
                diagnostic_explanation=(
                    "Process received SIGKILL (exit 137). Lacks explicit OOM confirmation; investigate host dmesg and orchestrator events."
                ),
            )

    @staticmethod
    def _evaluate_crash_loop(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Detect Kubernetes CrashLoopBackOff."""
        matched_refs: List[str] = []
        supporting: List[str] = []

        for k8s in evidence.k8s_events:
            msg = str(k8s.get("message") or "")
            reason = str(k8s.get("reason") or "")
            restarts = k8s.get("restarts", 0)
            if RE_CRASH_LOOP.search(msg) or RE_CRASH_LOOP.search(reason) or restarts >= 3:
                supporting.append(f"Pod reporting crash loop: {reason} ({restarts} restarts)")

        for eid, text in text_entries:
            if RE_CRASH_LOOP.search(text):
                matched_refs.append(eid)
                supporting.append(f"CrashLoopBackOff detected in evidence [{eid}]")

        if not (matched_refs or supporting):
            return None

        # Check for log snippet corroboration (e.g. fatal exception or traceback)
        has_app_traceback = any(
            "traceback" in s.lower() or "fatal" in s.lower() or "error" in s.lower()
            for s in evidence.log_snippets
        )
        corrob = 0.15 if has_app_traceback else 0.05
        if has_app_traceback:
            supporting.append(
                "Application logs contain fatal exceptions coinciding with startup crashes"
            )

        confidence = calculate_evidence_confidence(
            base_match=0.90,
            source_weight=0.95,
            corroboration=corrob,
            contradiction=0.00,
            explanation="CrashLoopBackOff detected with Kubernetes event status and restart counter",
        )
        return PatternMatchResult(
            pattern_id="K8S_CRASH_LOOP_BACKOFF",
            matched=True,
            title="Kubernetes Pod Entering CrashLoopBackOff",
            description="The pod container application terminates immediately upon launch and Kubernetes is backing off restarts.",
            category=Category.CONTAINER,
            severity=Severity.HIGH,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=[],
            missing_evidence=[],
            is_confirmed=confidence.final_score >= 0.85,
            diagnostic_explanation="Application crashes immediately after process execution; check entrypoint configuration and credentials.",
        )

    @staticmethod
    def _evaluate_image_pull(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Detect Kubernetes ImagePullBackOff / ErrImagePull."""
        matched_refs: List[str] = []
        supporting: List[str] = []

        for k8s in evidence.k8s_events:
            msg = str(k8s.get("message") or "")
            reason = str(k8s.get("reason") or "")
            if RE_IMAGE_PULL.search(msg) or RE_IMAGE_PULL.search(reason):
                supporting.append(f"K8s image failure event: {reason} - {msg}")

        for eid, text in text_entries:
            if RE_IMAGE_PULL.search(text):
                matched_refs.append(eid)
                supporting.append(f"Image pull error recorded in [{eid}]")

        if not (matched_refs or supporting):
            return None

        confidence = calculate_evidence_confidence(
            base_match=0.95,
            source_weight=0.95,
            corroboration=0.05,
            contradiction=0.00,
            explanation="ImagePullBackOff detected via orchestrator event and registry error",
        )
        return PatternMatchResult(
            pattern_id="K8S_IMAGE_PULL_BACKOFF",
            matched=True,
            title="Kubernetes ImagePullBackOff or Image Pull Failure",
            description="Kubernetes cannot pull the specified container image from the container registry.",
            category=Category.CONTAINER,
            severity=Severity.HIGH,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=[],
            missing_evidence=[],
            is_confirmed=True,
            diagnostic_explanation="Registry authentication failed or image tag does not exist.",
        )

    @staticmethod
    def _evaluate_db_connection(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Detect database ConnectionRefused failure pattern."""
        matched_refs: List[str] = []
        supporting: List[str] = []
        contradicting: List[str] = []

        for eid, text in text_entries:
            if RE_DB_CONN_REFUSED.search(text):
                matched_refs.append(eid)
                supporting.append(f"Connection refused pattern matched in [{eid}]")

        if not matched_refs:
            return None

        # Check for corroborating probe failure
        has_probe_failure = len(evidence.probe_failures) > 0
        for pf in evidence.probe_failures:
            status_code = pf.get("status_code", 200)
            if status_code in (500, 502, 503):
                supporting.append(f"Corroborating health probe failure HTTP {status_code}")

        # Check for contradictory probe success (e.g. database health probe is actually 200 OK)
        has_probe_ok = False
        for a in alerts:
            if a.source.value == "HEALTH_PROBE" and a.raw_data.get("status_code") == 200:
                has_probe_ok = True
                contradicting.append(
                    "Health probe endpoint reports HTTP 200 OK for database component"
                )

        corrob = 0.15 if has_probe_failure else 0.00
        contra = 0.40 if has_probe_ok else 0.00

        confidence = calculate_evidence_confidence(
            base_match=0.80,
            source_weight=0.85,
            corroboration=corrob,
            contradiction=contra,
            explanation="Database connection refusal evaluated with log patterns and probe status",
        )
        return PatternMatchResult(
            pattern_id="DB_CONNECTION_REFUSED",
            matched=True,
            title="Database Connection Refused (ECONNREFUSED)",
            description="The application failed to establish a network connection to the target database instance.",
            category=Category.APPLICATION,
            severity=Severity.HIGH,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_evidence=[],
            is_confirmed=confidence.final_score >= 0.85 and contra == 0.0,
            diagnostic_explanation="Database service daemon is stopped, unreachable over network, or port is blocked by firewall.",
        )

    @staticmethod
    def _evaluate_cpu_throttling(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Detect CPU saturation and CFS cgroup throttling."""
        cpu_val = 0.0
        matched_refs: List[str] = []
        supporting: List[str] = []
        contradicting: List[str] = []

        for comp, val in evidence.metric_anomalies.items():
            if "cpu" in comp.lower():
                cpu_val = max(cpu_val, val)

        for a in alerts:
            if a.source.value == "METRICS":
                if "cpu_percent" in a.raw_data:
                    pct = float(a.raw_data.get("cpu_percent", 0.0))
                    if pct >= 85.0:
                        cpu_val = max(cpu_val, pct)
                        matched_refs.append(f"EVID-ALERT-{a.alert_id}")
                elif "cpu" in a.component.lower():
                    val = evidence.metric_anomalies.get(a.component, 0.0)
                    if val >= 85.0:
                        cpu_val = max(cpu_val, val)
                        matched_refs.append(f"EVID-ALERT-{a.alert_id}")

        for eid, text in text_entries:
            if "cpu throttle" in text.lower() or "cfs quota" in text.lower():
                matched_refs.append(eid)
                supporting.append(f"CPU throttling message found in [{eid}]")

        if cpu_val < 85.0 and not matched_refs:
            return None

        if cpu_val >= 85.0:
            supporting.append(f"Recorded CPU utilization at {cpu_val:.1f}% (threshold 85%)")

        # Contradiction: explicit CPU metric indicating low utilization (< 40%)
        if cpu_val > 0 and cpu_val < 40.0:
            contradicting.append(f"Metrics report normal/low CPU utilization at {cpu_val:.1f}%")

        corrob = 0.10 if any(RE_UPSTREAM_TIMEOUT.search(t) for _, t in text_entries) else 0.00
        contra = 0.40 if contradicting else 0.00

        confidence = calculate_evidence_confidence(
            base_match=0.80,
            source_weight=0.90,
            corroboration=corrob,
            contradiction=contra,
            explanation="CPU throttling evaluated from Prometheus metrics and cgroup events",
        )
        return PatternMatchResult(
            pattern_id="CPU_THROTTLING",
            matched=True,
            title="CPU Saturation and CFS Throttling",
            description="CPU consumption exceeded allocation limits, resulting in CFS throttling and high processing latency.",
            category=Category.INFRASTRUCTURE,
            severity=Severity.MEDIUM,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_evidence=[],
            is_confirmed=confidence.final_score >= 0.85,
            diagnostic_explanation="Compute demand exceeds assigned CPU limits or shares; scale up core allocation.",
        )

    @staticmethod
    def _evaluate_memory_exhaustion(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
        oom_result: Optional[PatternMatchResult],
    ) -> Optional[PatternMatchResult]:
        """Detect memory exhaustion when NO confirmed OOM kill occurred."""
        # If confirmed OOM already matched, OOMKilled takes precedence
        if oom_result and oom_result.pattern_id == "CONTAINER_OOM_KILLED":
            return None

        mem_val = 0.0
        matched_refs: List[str] = []
        supporting: List[str] = []

        for comp, val in evidence.metric_anomalies.items():
            if "memory" in comp.lower() or "mem" in comp.lower():
                mem_val = max(mem_val, val)

        for a in alerts:
            if a.source.value == "METRICS":
                pct = float(a.raw_data.get("memory_percent", 0.0))
                if pct >= 85.0:
                    mem_val = max(mem_val, pct)
                    matched_refs.append(f"EVID-ALERT-{a.alert_id}")

        for eid, text in text_entries:
            if "memory exhaustion" in text.lower() or "high memory" in text.lower():
                matched_refs.append(eid)
                supporting.append(f"Memory warning recorded in [{eid}]")

        if mem_val < 85.0 and not matched_refs:
            return None

        if mem_val >= 85.0:
            supporting.append(
                f"Memory utilization at {mem_val:.1f}% exceeds safe operational threshold (85%)"
            )

        confidence = calculate_evidence_confidence(
            base_match=0.80,
            source_weight=0.90,
            corroboration=0.00,
            contradiction=0.00,
            explanation="High memory utilization without confirmed OOM kill",
        )
        return PatternMatchResult(
            pattern_id="MEMORY_EXHAUSTION",
            matched=True,
            title="High Memory Utilization / Approaching Saturation",
            description=(
                "Host or container memory utilization is critically high (>85%). Note: High memory alone "
                "does NOT confirm an OOM kill; monitoring indicates starvation risk without kernel termination."
            ),
            category=Category.INFRASTRUCTURE,
            severity=Severity.MEDIUM,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=[],
            missing_evidence=["Kernel OOM killer log", "Process termination event"],
            is_confirmed=confidence.final_score >= 0.85,
            diagnostic_explanation="High memory consumption indicates a leak, growing cache, or undersized provisioning.",
        )

    @staticmethod
    def _evaluate_readiness_probe(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Detect readiness probe failures causing cascading service unavailability."""
        matched_refs: List[str] = []
        supporting: List[str] = []
        contradicting: List[str] = []

        for pf in evidence.probe_failures:
            status = pf.get("status_code", 200)
            if status in (500, 502, 503):
                supporting.append(
                    f"Health probe check returned HTTP {status} ({pf.get('checks', {})})"
                )

        for eid, text in text_entries:
            if (
                "503 service unavailable" in text.lower()
                or "readiness probe failed" in text.lower()
            ):
                matched_refs.append(eid)
                supporting.append(f"Readiness probe failure logged in [{eid}]")

        if not (matched_refs or supporting):
            return None

        # Check for contradictory subsequent healthy probe
        for a in alerts:
            if a.source.value == "HEALTH_PROBE" and a.raw_data.get("status_code") == 200:
                contradicting.append("Subsequent health probe succeeded with HTTP 200 OK")

        corrob = 0.15 if len(supporting) >= 2 else 0.05
        contra = 0.40 if contradicting else 0.00

        confidence = calculate_evidence_confidence(
            base_match=0.85,
            source_weight=0.95,
            corroboration=corrob,
            contradiction=contra,
            explanation="Readiness probe failure evaluated from HTTP probe adapter events",
        )
        return PatternMatchResult(
            pattern_id="READINESS_PROBE_FAILURE",
            matched=True,
            title="Application Readiness Probe Failure (HTTP 503)",
            description="The application readiness endpoint (/readyz) reported failure, removing the instance from active service endpoints.",
            category=Category.APPLICATION,
            severity=Severity.HIGH,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_evidence=[],
            is_confirmed=confidence.final_score >= 0.85 and contra == 0.0,
            diagnostic_explanation="Internal service dependencies failed readiness evaluation; ingress traffic has been cut off.",
        )

    @staticmethod
    def _evaluate_upstream_timeout(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Detect upstream dependency request timeouts."""
        matched_refs: List[str] = []
        supporting: List[str] = []

        for eid, text in text_entries:
            if RE_UPSTREAM_TIMEOUT.search(text):
                matched_refs.append(eid)
                supporting.append(f"Request timeout pattern matched in [{eid}]")

        if not matched_refs:
            return None

        confidence = calculate_evidence_confidence(
            base_match=0.75,
            source_weight=0.85,
            corroboration=0.00,
            contradiction=0.00,
            explanation="Upstream request timeout detected in application log traces",
        )
        return PatternMatchResult(
            pattern_id="UPSTREAM_REQUEST_TIMEOUT",
            matched=True,
            title="Upstream Dependency Request Timeout",
            description="Outbound HTTP or RPC requests to an upstream microservice timed out, creating downstream latency pileups.",
            category=Category.NETWORK,
            severity=Severity.MEDIUM,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=[],
            missing_evidence=[],
            is_confirmed=confidence.final_score >= 0.85,
            diagnostic_explanation="Upstream service is unresponsive or latency exceeds client socket timeout.",
        )

    @staticmethod
    def _evaluate_llm_rate_limiting(
        alerts: List[Alert],
        evidence: CorrelatedEvidence,
        text_entries: List[tuple[str, str]],
    ) -> Optional[PatternMatchResult]:
        """Detect LLM provider rate limiting (HTTP 429 / Quota Exhaustion)."""
        matched_refs: List[str] = []
        supporting: List[str] = []

        for eid, text in text_entries:
            if RE_LLM_RATE_LIMIT.search(text):
                matched_refs.append(eid)
                supporting.append(f"Rate limiting error pattern matched in [{eid}]")

        for a in alerts:
            if a.source.value == "LOGS" and RE_LLM_RATE_LIMIT.search(a.message):
                matched_refs.append(f"EVID-ALERT-{a.alert_id}")
                supporting.append(f"LLM provider returned rate limit / quota error: {a.message}")

        if not matched_refs:
            return None

        confidence = calculate_evidence_confidence(
            base_match=0.90,
            source_weight=0.90,
            corroboration=0.05,
            contradiction=0.00,
            explanation="LLM provider rate limit (HTTP 429) verified in provider response logs",
        )
        return PatternMatchResult(
            pattern_id="LLM_RATE_LIMITING",
            matched=True,
            title="LLM Provider Rate Limiting / Quota Exhaustion (HTTP 429)",
            description="The AI provider rejected generation requests due to rate limits (RPM/TPM) or billing quota exhaustion.",
            category=Category.PROVIDER,
            severity=Severity.MEDIUM,
            confidence_breakdown=confidence,
            evidence_refs=list(dict.fromkeys(matched_refs)),
            supporting_evidence=supporting,
            contradicting_evidence=[],
            missing_evidence=[],
            is_confirmed=True,
            diagnostic_explanation="Provider request rate exceeded allowed threshold; requests must back off or fallback provider engaged.",
        )
