# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Abstract base agent establishing execution lifecycle, safety guard rails, and LLMRouter integration."""

from abc import ABC, abstractmethod
import re
import time
from typing import Any, List, Optional
import uuid

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
from devops_ai.core.config import Settings, get_settings
from devops_ai.core.exceptions import (
    AgentError,
    AgentLimitExceededError,
    AgentSecurityError,
)
from devops_ai.core.logging import get_logger, scrub_secrets
from devops_ai.llm.router import GenerationResult, LLMRouter

logger = get_logger("agents.base")

# Prohibited destructive or mutating commands for agent safety enforcement
_RE_PROHIBITED_MUTATIONS = re.compile(
    r"(?i)\b("
    r"rm\s+-rf|"
    r"drop\s+table|"
    r"delete\s+from|"
    r"truncate\s+table|"
    r"kubectl\s+delete|"
    r"kubectl\s+drain|"
    r"docker\s+rm|"
    r"docker\s+stop|"
    r"docker\s+kill|"
    r"kill\s+-9|"
    r"mkfs|"
    r"shutdown|"
    r"reboot"
    r")\b"
)


def _escape_untrusted_evidence(text: str) -> str:
    """Neutralize XML delimiter breakout attempts in untrusted inputs and apply entity escaping."""
    if not text:
        return ""
    sanitized = re.sub(
        r"(?i)</?\s*(?:UNTRUSTED_AGENT_INPUT|SYSTEM_MANDATE|AGENT_CONTEXT|UNTRUSTED_FINOPS_EVIDENCE)[^>]*>",
        "[DELIMITER_TAG_NEUTRALIZED]",
        text,
    )
    return (
        sanitized.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


class BaseAgent(ABC):
    """Abstract base class establishing the contract and lifecycle for all AI agents."""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        description: str,
        capabilities: Optional[List[AgentCapability]] = None,
        limits: Optional[AgentLimits] = None,
        settings: Optional[Settings] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.agent_id = f"AGT-{role.value[:4]}-{uuid.uuid4().hex[:6].upper()}"
        self.name = name
        self.role = role
        self.description = description
        self.capabilities = capabilities or [AgentCapability.SYNTHESIS]
        self.status = AgentStatus.IDLE
        self.limits = limits or AgentLimits(
            max_iterations=self.settings.agent_max_iterations,
            timeout_seconds=self.settings.agent_timeout_seconds,
            max_input_chars=self.settings.agent_max_input_chars,
            read_only=self.settings.agent_read_only_mode,
        )
        self.provider_name = provider_name or self.settings.default_llm_provider
        self.model = model
        self.logger = get_logger(f"agents.{self.name}")

    def initialize(self) -> None:
        """Initialize agent resources and verify read-only configuration."""
        self.logger.info(
            f"Initializing agent '{self.name}' ({self.agent_id}) with role '{self.role.value}'"
        )
        self.status = AgentStatus.INITIALIZING
        # Verify read-only safety boundary
        if not self.limits.read_only:
            self.status = AgentStatus.FAILED
            raise AgentSecurityError(
                f"Agent '{self.name}' cannot be initialized: read_only=False is strictly forbidden"
            )
        self.status = AgentStatus.IDLE

    def _check_safety_guardrails(self, task: AgentTask) -> None:
        """Validate safety constraints before executing task."""
        # 1. Enforce read-only constraint
        if not self.limits.read_only or not task.read_only:
            raise AgentSecurityError(
                f"Agent '{self.name}' operates in strict read-only mode. Mutating tasks are prohibited."
            )

        # 2. Check for prohibited destructive commands in task description
        if _RE_PROHIBITED_MUTATIONS.search(task.description):
            raise AgentSecurityError(
                f"Task '{task.task_id}' contains prohibited destructive/mutating command patterns: "
                f"'{task.description[:100]}'"
            )

    def _sanitize_untrusted_input(self, raw_input: Any) -> str:
        """Sanitize, scrub secrets, and enforce character length bounds on untrusted input."""
        if raw_input is None:
            return ""
        text = str(raw_input)
        if len(text) > self.limits.max_input_chars:
            text = (
                text[: self.limits.max_input_chars]
                + "\n[TRUNCATED: Input exceeded agent character limit]"
            )
        scrubbed = scrub_secrets(text)
        return _escape_untrusted_evidence(scrubbed)

    def _generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> GenerationResult:
        """Invoke LLMRouter with provider fallback, timeout bounds, and secret scrubbing."""
        effective_timeout = timeout or self.limits.timeout_seconds
        return LLMRouter.generate_with_fallback(
            prompt=prompt,
            system_prompt=system_prompt,
            provider_name=self.provider_name,
            model=self.model,
            timeout=effective_timeout,
            temperature=temperature,
            **kwargs,
        )

    def execute(
        self,
        task: AgentTask,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Execute a diagnostic or reasoning task through the standardized agent lifecycle."""
        start_time = time.monotonic()
        effective_timeout = task.timeout or self.limits.timeout_seconds
        ctx = context or AgentContext()
        steps: List[AgentStep] = []

        self.logger.info(
            f"Agent '{self.name}' ({self.agent_id}) accepted task '{task.task_id}': "
            f"'{task.description[:80]}' (timeout: {effective_timeout}s, max_iterations: {self.limits.max_iterations})"
        )

        try:
            # 1. Safety Guard Rails Verification
            self._check_safety_guardrails(task)

            # 2. State Transition: PLANNING
            self.status = AgentStatus.PLANNING
            plan_items = self.plan(task, ctx)
            self.logger.debug(f"Task '{task.task_id}' planned {len(plan_items)} action items")

            # 3. State Transition: EXECUTING
            self.status = AgentStatus.EXECUTING
            iteration = 0

            while iteration < self.limits.max_iterations:
                elapsed = time.monotonic() - start_time
                if elapsed > effective_timeout:
                    raise AgentLimitExceededError(
                        f"Task '{task.task_id}' exceeded timeout of {effective_timeout}s (elapsed: {elapsed:.2f}s)",
                        limit_type="timeout",
                    )

                iteration += 1
                step_obj = self.step(task, ctx, iteration)
                steps.append(step_obj)
                ctx.history.append(step_obj)

                # Check if step indicates completion
                if step_obj.action.strip().upper() in ("COMPLETE", "STOP", "FINISHED", "DONE"):
                    break

            if iteration >= self.limits.max_iterations and self.status == AgentStatus.EXECUTING:
                self.logger.warning(
                    f"Task '{task.task_id}' reached maximum iterations ({self.limits.max_iterations})"
                )

            # 4. State Transition: COMPLETED
            result = self.synthesize(task, ctx)
            result.steps = steps
            result.duration_seconds = round(time.monotonic() - start_time, 3)
            self.status = AgentStatus.COMPLETED

            self.logger.info(
                f"Agent '{self.name}' completed task '{task.task_id}' in {result.duration_seconds}s "
                f"across {len(steps)} step(s)"
            )
            return result

        except AgentLimitExceededError as e:
            duration = round(time.monotonic() - start_time, 3)
            self.status = AgentStatus.FAILED
            self.logger.warning(f"Agent '{self.name}' limit exceeded on task '{task.task_id}': {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                summary=f"Task aborted: {e.message}",
                error=e.message,
                steps=steps,
                duration_seconds=duration,
                metadata={"limit_type": e.limit_type},
            )

        except AgentSecurityError as e:
            duration = round(time.monotonic() - start_time, 3)
            self.status = AgentStatus.FAILED
            self.logger.error(
                f"Agent '{self.name}' security boundary triggered on task '{task.task_id}': {e}"
            )
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                summary=f"Security violation: {e.message}",
                error=e.message,
                steps=steps,
                duration_seconds=duration,
                metadata={"security_violation": True},
            )

        except AgentError as e:
            duration = round(time.monotonic() - start_time, 3)
            self.status = AgentStatus.FAILED
            self.logger.error(f"Agent '{self.name}' failed on task '{task.task_id}': {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                summary=f"Agent error: {e.message}",
                error=e.message,
                steps=steps,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = round(time.monotonic() - start_time, 3)
            self.status = AgentStatus.FAILED
            sanitized = scrub_secrets(str(e))
            self.logger.error(
                f"Unexpected failure executing task '{task.task_id}' on agent '{self.name}': {sanitized}"
            )
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                summary=f"Unexpected execution failure: {sanitized}",
                error=sanitized,
                steps=steps,
                duration_seconds=duration,
            )

    def cleanup(self) -> None:
        """Reset agent to idle state."""
        self.status = AgentStatus.IDLE

    def cancel(self) -> None:
        """Cancel ongoing execution."""
        self.status = AgentStatus.CANCELLED

    @abstractmethod
    def plan(self, task: AgentTask, context: AgentContext) -> List[str]:
        """Formulate high-level sequence of diagnostic reasoning items for the task."""

    @abstractmethod
    def step(self, task: AgentTask, context: AgentContext, step_num: int) -> AgentStep:
        """Execute a single iterative reasoning or diagnostic inquiry step."""

    @abstractmethod
    def synthesize(self, task: AgentTask, context: AgentContext) -> AgentResult:
        """Synthesize accumulated observations and steps into structured findings and summary."""
