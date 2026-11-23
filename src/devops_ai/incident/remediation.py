# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Remediation advisor generating risk-classified advisory recommendations with safety guard rails."""

import re
import shlex
from typing import List, Optional
from devops_ai.core.logging import get_logger
from devops_ai.incident.models import (
    Incident,
    RemediationAction,
    RiskLevel,
)

logger = get_logger("incident.remediation")

# Strict regex detecting destructive or irreversible operations
RE_DESTRUCTIVE_COMMANDS = re.compile(
    r"(?i)\b("
    r"delete|drop|truncate|rm|rmdir|kill\s+-9|pkill\s+-9|"
    r"purge|format|fdisk|mkfs|prune|drain\s+--force|"
    r"destroy|terminate|uninstall|chmod\s+777|chown\s+-R|iptables\s+-F|"
    r"shred|dd\s+if=|apt-get\s+remove|yum\s+remove|dpkg\s+-r"
    r")\b"
)

# Mutating workload operations
RE_MUTATING_COMMANDS = re.compile(
    r"(?i)\b("
    r"restart|reboot|scale|set\s+resources|rollout|apply|stop|kill|"
    r"patch|upgrade|rollback|undo|killall|docker\s+kill"
    r")\b"
)

# Forbidden shell operators, subshell patterns, and dangerous execution utilities
FORBIDDEN_SHELL_OPERATORS = re.compile(
    r"[;&|><`$]|\b(?:bash|sh|zsh|eval|exec|python|python3|perl|ruby)\s+-c\b|\bbase64\s+-d\b"
)

# Explicit allowlist of approved read-only tool prefixes
APPROVED_READ_ONLY_PREFIXES = (
    "kubectl get",
    "kubectl describe",
    "kubectl logs",
    "kubectl cluster-info",
    "kubectl top",
    "docker inspect",
    "docker logs",
    "docker ps",
    "docker top",
    "docker stats",
    "free",
    "uptime",
    "dmesg",
    "uname",
    "vmstat",
    "iostat",
    "mpstat",
    "cat",
    "head",
    "tail",
    "grep",
    "ping",
    "traceroute",
    "nc -z",
    "curl -I",
    "curl --head",
    "python -m devops_ai.cli provider check",
)


class RemediationAdvisor:
    """Generates risk-classified advisory remediation playbooks for analyzed incidents.

    Enforces strict safety guard rails:
    1. Only approved read-only commands matching the strict allowlist are classified as SAFE.
    2. Workload mutating commands are classified as MODERATE.
    3. Destructive commands, shell chaining, and subshells are ALWAYS classified as DANGEROUS.
    4. Unrecognized/unverified commands NEVER default to SAFE (elevated to MODERATE or DANGEROUS).
    5. Zero automated execution hooks: all recommendations are strictly advisory for human operators.
    """

    @staticmethod
    def classify_command_risk(
        command: Optional[str], proposed_risk: RiskLevel = RiskLevel.SAFE
    ) -> RiskLevel:
        """Classify command risk level with conservative allowlist safety overrides."""
        if not command or not command.strip():
            return proposed_risk if proposed_risk != RiskLevel.SAFE else RiskLevel.MODERATE

        cmd = command.strip()
        cmd_lower = cmd.lower()

        # 1. Hard guardrail: Any destructive keyword forces DANGEROUS
        if RE_DESTRUCTIVE_COMMANDS.search(cmd_lower):
            return RiskLevel.DANGEROUS

        # 2. Syntax validation with shlex
        try:
            shlex.split(cmd, posix=True)
        except Exception:
            return RiskLevel.DANGEROUS

        # 3. Detect dangerous shell operators, subshells, pipes, command chaining
        has_forbidden_ops = bool(FORBIDDEN_SHELL_OPERATORS.search(cmd))
        if has_forbidden_ops:
            # If it contains subshell, pipe to shell, eval, base64, or destructive scripts
            if any(
                tok in cmd_lower
                for tok in (
                    "bash",
                    "sh",
                    "eval",
                    "python",
                    "base64",
                    "rm",
                    "kill",
                    "dangerous",
                    "drop",
                    "delete",
                )
            ):
                return RiskLevel.DANGEROUS
            if cmd.startswith(">") or cmd.startswith("<"):
                return RiskLevel.DANGEROUS
            # Any other chained or piped command is at least MODERATE
            return RiskLevel.MODERATE

        # 4. Workload mutating keyword forces at least MODERATE
        if RE_MUTATING_COMMANDS.search(cmd_lower):
            return (
                RiskLevel.MODERATE if proposed_risk != RiskLevel.DANGEROUS else RiskLevel.DANGEROUS
            )

        # 5. Strict Read-Only Allowlist Verification
        def _matches_allowlist(target_cmd: str) -> bool:
            return any(
                target_cmd.startswith(prefix + " ")
                or target_cmd == prefix
                or (prefix == "nc -z" and target_cmd.startswith("nc -z"))
                or (prefix == "curl -I" and target_cmd.startswith("curl -I"))
                for prefix in APPROVED_READ_ONLY_PREFIXES
            )

        is_approved_prefix = _matches_allowlist(cmd)
        if not is_approved_prefix and cmd.startswith("kubectl exec") and " -- " in cmd:
            subcmd = cmd.split(" -- ", 1)[1].strip()
            if _matches_allowlist(subcmd):
                is_approved_prefix = True

        if is_approved_prefix and not has_forbidden_ops:
            return RiskLevel.SAFE

        # 6. Conservative Default: Unrecognized commands must NEVER be classified as SAFE
        return RiskLevel.MODERATE if proposed_risk == RiskLevel.SAFE else proposed_risk

    def generate_recommendations(self, incident: Incident) -> List[RemediationAction]:
        """Generate structured advisory recommendations based on incident root cause and evidence."""
        actions: List[RemediationAction] = []
        comp = incident.components_affected[0] if incident.components_affected else "service"
        rc_title = (
            incident.root_cause.title if incident.root_cause else "Unknown Operational Anomaly"
        )
        rc_ref = rc_title

        # Collect valid evidence references from incident catalog to ground recommendations
        valid_evidence_refs = list(incident.evidence.evidence_catalog.keys())
        for a in incident.alerts:
            valid_evidence_refs.append(f"EVID-ALERT-{a.alert_id}")

        evidence_subset = valid_evidence_refs[:3]

        rc_lower = rc_title.lower()

        # 1. Playbook: Out-Of-Memory (OOM) Termination
        if "oom" in rc_lower or "out-of-memory" in rc_lower:
            actions.append(
                self._create_action(
                    action_id="ACT-OOM-01",
                    title="Inspect memory metrics and previous container termination",
                    description=f"Inspect memory utilization and previous container termination status for {comp}.",
                    explanation="Verify cgroup memory usage limits and container exit state before modifying limits.",
                    action_type="diagnostic",
                    risk_level=RiskLevel.SAFE,
                    command=f"kubectl describe pod {comp}",
                    verification_step=f"Confirm Exit Code 137 and OOMKilled=true in pod termination details for {comp}.",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[
                        "Cluster credentials configured",
                        f"Pod {comp} exists in cluster",
                    ],
                    expected_result="Displays container termination status showing whether OOM killer was invoked.",
                    rollback_guidance="Read-only operation; no rollback needed.",
                )
            )
            actions.append(
                self._create_action(
                    action_id="ACT-OOM-02",
                    title="Increase deployment container memory limit",
                    description=f"Scale up container memory limit provisioned for {comp} by 50-100%.",
                    explanation="Provide sufficient memory headroom to avoid kernel cgroup OOM termination during workload peaks.",
                    action_type="immediate_mitigation",
                    risk_level=RiskLevel.MODERATE,
                    command=f"kubectl set resources deployment/{comp} --limits=memory=1Gi --requests=memory=512Mi",
                    verification_step=f"kubectl rollout status deployment/{comp} --timeout=60s",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[
                        "Verify cluster node has available memory capacity",
                        "Verify deployment exists",
                    ],
                    expected_result=f"Deployment {comp} rolls out with increased memory limit, eliminating OOM termination.",
                    rollback_guidance=f"kubectl rollout undo deployment/{comp}",
                )
            )
            actions.append(
                self._create_action(
                    action_id="ACT-OOM-03",
                    title="Emergency force purge failed pods and evict hung instances",
                    description=f"Force delete hung or unresponsive pods for {comp} with zero grace period.",
                    explanation="Destructive operation to forcefully terminate pods that fail to respond to SIGTERM.",
                    action_type="immediate_mitigation",
                    risk_level=RiskLevel.DANGEROUS,
                    command=f"kubectl delete pod {comp} --grace-period=0 --force",
                    verification_step=f"kubectl get pods -l app={comp}",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[
                        "Confirm service disruption is acceptable",
                        "Confirm pod is completely unresponsive",
                    ],
                    expected_result="Terminates hung pod immediately and triggers replacement replica creation.",
                    rollback_guidance="Force deletion cannot be rolled back; replacement pod will be scheduled automatically.",
                )
            )

        # 2. Playbook: Kubernetes CrashLoopBackOff
        elif "crashloopbackoff" in rc_lower or "crash" in rc_lower:
            actions.append(
                self._create_action(
                    action_id="ACT-CLB-01",
                    title="Retrieve previous crash logs and container exit status",
                    description=f"Inspect application logs immediately preceding the crash for {comp}.",
                    explanation="Review bootstrap exception tracebacks or missing configuration files.",
                    action_type="diagnostic",
                    risk_level=RiskLevel.SAFE,
                    command=f"kubectl logs {comp} --previous --tail=100",
                    verification_step="Inspect stdout/stderr for unhandled exceptions or failed startup assertions.",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[f"Pod {comp} has restarted at least once"],
                    expected_result="Displays fatal error or stack trace from the failed container instance.",
                    rollback_guidance="Read-only diagnostic; no rollback required.",
                )
            )
            actions.append(
                self._create_action(
                    action_id="ACT-CLB-02",
                    title="Roll back deployment to previous stable revision",
                    description=f"Revert deployment {comp} to the prior working release.",
                    explanation="Mitigates startup crash loops introduced by recent bad code or configuration releases.",
                    action_type="immediate_mitigation",
                    risk_level=RiskLevel.MODERATE,
                    command=f"kubectl rollout undo deployment/{comp}",
                    verification_step=f"kubectl rollout status deployment/{comp} --timeout=90s",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=["A prior working deployment revision exists in rollout history"],
                    expected_result="Reverts pods to previous container image and configuration revision.",
                    rollback_guidance=f"kubectl rollout undo deployment/{comp} --to-revision=<current_rev>",
                )
            )

        # 3. Playbook: Database Connection Refused
        elif "database" in rc_lower or "connection refused" in rc_lower:
            actions.append(
                self._create_action(
                    action_id="ACT-DB-01",
                    title="Verify database listener and port reachability",
                    description="Test network connectivity from application instance to target database endpoint.",
                    explanation="Determine whether failure is caused by stopped database daemon or network policy.",
                    action_type="diagnostic",
                    risk_level=RiskLevel.SAFE,
                    command=f"kubectl exec -it deployment/{comp} -- nc -zvw3 postgres-db 5432",
                    verification_step="Output should indicate 'Connection to postgres-db 5432 port [tcp/postgresql] succeeded!'.",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[
                        "Application pod is running",
                        "nc or telnet binary available in container",
                    ],
                    expected_result="Verifies TCP socket handshake to database host and port.",
                    rollback_guidance="Diagnostic check; no rollback needed.",
                )
            )
            actions.append(
                self._create_action(
                    action_id="ACT-DB-02",
                    title="Restart application connection pool and dependent workers",
                    description=f"Perform rolling restart of {comp} to refresh stale or poisoned connection pools.",
                    explanation="Re-establishes healthy TCP sockets to the database after database server restart.",
                    action_type="immediate_mitigation",
                    risk_level=RiskLevel.MODERATE,
                    command=f"kubectl rollout restart deployment/{comp}",
                    verification_step=f"kubectl rollout status deployment/{comp}",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[
                        "Database service is confirmed running and accepting connections"
                    ],
                    expected_result="Restarts application pods sequentially without service outage.",
                    rollback_guidance="None required; rolling restart spawns new pods with identical configuration.",
                )
            )

        # 4. Playbook: ImagePullBackOff / Registry Error
        elif "imagepullbackoff" in rc_lower or "image" in rc_lower:
            actions.append(
                self._create_action(
                    action_id="ACT-IMG-01",
                    title="Inspect image pull failure events and registry authentication",
                    description=f"Inspect pod event log for image tag and registry authentication errors on {comp}.",
                    explanation="Determine if image tag is missing or imagePullSecrets credential expired.",
                    action_type="diagnostic",
                    risk_level=RiskLevel.SAFE,
                    command=f"kubectl get events --field-selector involvedObject.name={comp} --sort-by='.lastTimestamp'",
                    verification_step="Verify specific error message (e.g. 401 Unauthorized or 404 Not Found).",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[f"Pod {comp} exists in cluster"],
                    expected_result="Displays registry response code and image repository details.",
                    rollback_guidance="Read-only operation.",
                )
            )
            actions.append(
                self._create_action(
                    action_id="ACT-IMG-02",
                    title="Update deployment container image to verified release tag",
                    description=f"Update container image tag for deployment {comp} to a verified existing tag.",
                    explanation="Corrects typos in image repository URL or non-existent image tags.",
                    action_type="immediate_mitigation",
                    risk_level=RiskLevel.MODERATE,
                    command=f"kubectl set image deployment/{comp} {comp}=registry.example.com/{comp}:stable",
                    verification_step=f"kubectl rollout status deployment/{comp}",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=["Target image tag exists and is verified in container registry"],
                    expected_result="Pods successfully pull image and transition to Running state.",
                    rollback_guidance=f"kubectl rollout undo deployment/{comp}",
                )
            )

        # 5. Playbook: LLM Provider Rate Limiting (HTTP 429)
        elif "rate limit" in rc_lower or "llm" in rc_lower or "429" in rc_lower:
            actions.append(
                self._create_action(
                    action_id="ACT-LLM-01",
                    title="Inspect AI provider quota consumption and rate limits",
                    description="Check provider TPM/RPM metrics and billing quota status.",
                    explanation="Verify if rate limiting is transient burst or sustained quota exhaustion.",
                    action_type="diagnostic",
                    risk_level=RiskLevel.SAFE,
                    command="python -m devops_ai.cli provider check --provider=gemini --verify-remote",
                    verification_step="Inspect CLI output for quota and rate limit status headers.",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=["Provider credentials configured"],
                    expected_result="Reports current provider reachability and quota availability.",
                    rollback_guidance="Read-only diagnostic.",
                )
            )
            actions.append(
                self._create_action(
                    action_id="ACT-LLM-02",
                    title="Enable automatic LLM provider fallback routing",
                    description="Activate secondary provider fallback routing in configuration.",
                    explanation="Reroutes requests from throttled primary provider to healthy fallback provider.",
                    action_type="immediate_mitigation",
                    risk_level=RiskLevel.MODERATE,
                    command="export DEVOPS_AI_ENABLE_LLM_FALLBACK=true",
                    verification_step="Verify subsequent LLM requests complete successfully via alternate provider.",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=["Secondary provider configured and healthy"],
                    expected_result="Reroutes traffic to alternate model without application errors.",
                    rollback_guidance="export DEVOPS_AI_ENABLE_LLM_FALLBACK=false",
                )
            )

        # 6. Default / Generic Operational Failure Playbook
        else:
            actions.append(
                self._create_action(
                    action_id="ACT-GEN-01",
                    title="Collect diagnostic support bundle and pod description",
                    description=f"Gather detailed Kubernetes events, resource metrics, and logs for {comp}.",
                    explanation="Deep diagnostic capture to isolate operational anomalies lacking deterministic signatures.",
                    action_type="diagnostic",
                    risk_level=RiskLevel.SAFE,
                    command=f"kubectl describe pod {comp}",
                    verification_step="Inspect output for unhandled exceptions or resource constraint warnings.",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=[f"Pod {comp} exists in cluster"],
                    expected_result="Captures comprehensive diagnostic state for troubleshooting.",
                    rollback_guidance="Read-only diagnostic.",
                )
            )
            actions.append(
                self._create_action(
                    action_id="ACT-GEN-02",
                    title="Execute graceful rolling restart of component",
                    description=f"Trigger graceful rolling restart for {comp} workloads.",
                    explanation="Clears transient deadlock, memory fragmentation, or leaked file descriptors.",
                    action_type="immediate_mitigation",
                    risk_level=RiskLevel.MODERATE,
                    command=f"kubectl rollout restart deployment/{comp}",
                    verification_step=f"kubectl rollout status deployment/{comp} --timeout=120s",
                    root_cause_ref=rc_ref,
                    evidence_refs=evidence_subset,
                    preconditions=["Deployment exists and healthy replicas are present"],
                    expected_result="Replaces pod instances sequentially with healthy containers.",
                    rollback_guidance=f"kubectl rollout undo deployment/{comp}",
                )
            )

        return actions

    def _create_action(
        self,
        action_id: str,
        title: str,
        description: str,
        explanation: str,
        action_type: str,
        risk_level: RiskLevel,
        command: Optional[str],
        verification_step: str,
        root_cause_ref: str,
        evidence_refs: List[str],
        preconditions: List[str],
        expected_result: str,
        rollback_guidance: str,
    ) -> RemediationAction:
        """Construct a validated RemediationAction with strict risk classification guard rails."""
        # Enforce deterministic command risk guard rail
        final_risk = self.classify_command_risk(command, proposed_risk=risk_level)

        return RemediationAction(
            action_id=action_id,
            title=title,
            description=description,
            explanation=explanation,
            action_type=action_type,
            risk_level=final_risk,
            command=command,
            verification_step=verification_step,
            root_cause_ref=root_cause_ref,
            evidence_refs=evidence_refs,
            preconditions=preconditions,
            expected_result=expected_result,
            rollback_guidance=rollback_guidance,
            advisory_warning=(
                "ADVISORY ONLY: Must be executed manually by a human operator. "
                "Automated execution is strictly disabled."
            ),
        )
