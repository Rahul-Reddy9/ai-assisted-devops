# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Concrete diagnostic AI agent specialized in DevOps telemetry analysis and advisory synthesis."""

import json
import re
from typing import Any, Dict, List, Optional

from devops_ai.agents.base import BaseAgent
from devops_ai.agents.models import (
    AgentCapability,
    AgentContext,
    AgentLimits,
    AgentResult,
    AgentRole,
    AgentStatus,
    AgentStep,
    AgentTask,
)
from devops_ai.core.config import Settings
from devops_ai.core.exceptions import LLMProviderError
from devops_ai.core.logging import scrub_secrets


class DevOpsDiagnosticAgent(BaseAgent):
    """Production-grade SRE and DevOps diagnostic agent for telemetry and incident evaluation."""

    def __init__(
        self,
        name: str = "devops-diagnostician",
        description: Optional[str] = None,
        limits: Optional[AgentLimits] = None,
        settings: Optional[Settings] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        desc = description or (
            "Autonomous DevOps diagnostic agent specialized in operational telemetry analysis, "
            "log anomaly investigation, and advisory remediation synthesis."
        )
        caps = [
            AgentCapability.READ_TELEMETRY,
            AgentCapability.ANALYZE_LOGS,
            AgentCapability.INSPECT_WORKLOADS,
            AgentCapability.INCIDENT_DIAGNOSIS,
            AgentCapability.REMEDIATION_ADVISORY,
            AgentCapability.SYNTHESIS,
        ]
        super().__init__(
            name=name,
            role=AgentRole.DIAGNOSTICIAN,
            description=desc,
            capabilities=caps,
            limits=limits,
            settings=settings,
            provider_name=provider_name,
            model=model,
        )
        self._current_plan: List[str] = []
        self._findings: List[Dict[str, Any]] = []

    def plan(self, task: AgentTask, context: AgentContext) -> List[str]:
        """Formulate structured sequence of diagnostic steps for the given task."""
        self._findings = []

        # System prompt with strict read-only safety mandate
        system_prompt = (
            "You are a specialized DevOps Site Reliability Engineer AI.\n"
            "Formulate a concise step-by-step diagnostic plan for the operational task.\n"
            "MANDATORY SAFETY RULES:\n"
            "1. You operate in STRICT READ-ONLY mode. Never propose or execute mutating actions.\n"
            "2. Focus exclusively on diagnostic inspection, log analysis, and root cause evaluation.\n"
            "3. Return a clean bulleted list of 2 to 4 diagnostic steps."
        )

        sanitized_desc = self._sanitize_untrusted_input(task.description)
        context_preview = self._sanitize_untrusted_input(json.dumps(task.context))
        user_prompt = (
            f"<AGENT_TASK>\n{sanitized_desc}\n</AGENT_TASK>\n\n"
            f"<UNTRUSTED_AGENT_INPUT>\n{context_preview}\n</UNTRUSTED_AGENT_INPUT>\n\n"
            "Outline your diagnostic plan."
        )

        try:
            gen_result = self._generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )
            raw_lines = [
                re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
                for line in gen_result.content.strip().splitlines()
                if line.strip()
            ]
            self._current_plan = [line for line in raw_lines if len(line) > 5][:4]
            if not self._current_plan:
                self._current_plan = ["Inspect provided telemetry", "Analyze failure indicators"]
        except (LLMProviderError, Exception) as e:
            self.logger.warning(
                f"LLM plan generation unavailable ({e}); adopting deterministic fallback plan"
            )
            self._current_plan = [
                "Inspect error messages and timestamps in task context",
                "Correlate metric anomalies and container exit codes",
                "Synthesize diagnostic findings and advisory remediation",
            ]

        return self._current_plan

    def step(self, task: AgentTask, context: AgentContext, step_num: int) -> AgentStep:
        """Execute a single iterative reasoning or diagnostic inquiry step."""
        plan_desc = (
            self._current_plan[step_num - 1]
            if step_num <= len(self._current_plan)
            else "Complete diagnostic synthesis"
        )

        # Evaluate deterministic observations from context if available
        observation = ""
        context_data = task.context

        if step_num == 1:
            # Step 1: Telemetry & Error inspection
            errors = context_data.get("errors") or context_data.get("logs") or []
            if errors:
                observation = f"Identified {len(errors)} error events in context stream."
                self._findings.append(
                    {
                        "category": "LOGS",
                        "detail": scrub_secrets(str(errors[:2])),
                    }
                )
            else:
                observation = "No explicit error bursts found in raw logs."

        elif step_num == 2:
            # Step 2: Metric and state inspection
            metrics = context_data.get("metrics") or {}
            exit_code = context_data.get("exit_code")
            if exit_code == 137 or "oom" in str(context_data).lower():
                observation = "Process terminated with exit code 137 (SIGKILL/OOM)."
                self._findings.append(
                    {
                        "category": "CONTAINER",
                        "detail": "Exit code 137 indicates forced termination (likely OOMKilled).",
                    }
                )
            elif metrics:
                observation = f"Analyzed {len(metrics)} telemetry metrics."
                self._findings.append(
                    {
                        "category": "METRICS",
                        "detail": scrub_secrets(str(metrics)),
                    }
                )
            else:
                observation = "Telemetry values are within nominal thresholds."

        else:
            observation = "Diagnostic analysis completed."

        is_final = step_num >= len(self._current_plan) or step_num >= 3
        action_name = "COMPLETE" if is_final else f"EVALUATE_STEP_{step_num}"

        return AgentStep(
            step_number=step_num,
            thought=f"Executing planned step: {plan_desc}",
            action=action_name,
            observation=observation,
        )

    def synthesize(self, task: AgentTask, context: AgentContext) -> AgentResult:
        """Synthesize accumulated findings and steps into structured findings and summary."""
        sanitized_desc = self._sanitize_untrusted_input(task.description)
        findings_str = (
            json.dumps(self._findings) if self._findings else "No critical anomalies identified."
        )

        system_prompt = (
            "You are a specialized DevOps SRE AI synthesizing diagnostic findings.\n"
            "Provide a concise executive summary and advisory recommendation.\n"
            "MANDATE: All recommendations are ADVISORY ONLY. Never propose destructive commands."
        )
        user_prompt = (
            f"Task: {sanitized_desc}\nFindings: {findings_str}\nProvide an executive summary."
        )

        try:
            gen_result = self._generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )
            summary = gen_result.content.strip()
        except (LLMProviderError, Exception):
            summary = (
                f"Diagnostic review completed for task '{task.task_id}'. "
                f"Identified {len(self._findings)} findings across {len(context.history)} steps. "
                "Advisory: Inspect relevant workload limits and logs before restarting."
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            summary=summary,
            findings=self._findings,
            steps=list(context.history),
            duration_seconds=0.0,
            metadata={
                "read_only": True,
                "advisory_only": True,
                "findings_count": len(self._findings),
            },
        )
