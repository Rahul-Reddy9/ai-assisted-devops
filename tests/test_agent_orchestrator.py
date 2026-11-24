# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for AgentOrchestrator coordination, task routing, and sequential pipelines."""

from typing import List

import pytest

from devops_ai.agents.base import BaseAgent
from devops_ai.agents.models import (
    AgentCapability,
    AgentContext,
    AgentResult,
    AgentRole,
    AgentStatus,
    AgentStep,
    AgentTask,
)
from devops_ai.agents.orchestrator import AgentOrchestrator
from devops_ai.core.exceptions import AgentNotFoundError


class FirstPipelineAgent(BaseAgent):
    """Pipeline agent 1: extracts logs and sets facts in context."""

    def __init__(self, name="agent-one"):
        super().__init__(
            name=name,
            role=AgentRole.OBSERVER,
            description="Extracts log errors",
            capabilities=[AgentCapability.ANALYZE_LOGS],
        )

    def plan(self, task: AgentTask, context: AgentContext) -> List[str]:
        return ["Scan log errors"]

    def step(self, task: AgentTask, context: AgentContext, step_num: int) -> AgentStep:
        context.facts.append("Identified 5 HTTP 500 error bursts")
        return AgentStep(
            step_number=step_num,
            thought="Parsed error bursts",
            action="COMPLETE",
            observation="5 error bursts recorded",
        )

    def synthesize(self, task: AgentTask, context: AgentContext) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            summary="Found 5 HTTP 500 errors",
            findings=[{"count": 5, "type": "HTTP_500"}],
            duration_seconds=0.05,
        )


class SecondPipelineAgent(BaseAgent):
    """Pipeline agent 2: consumes facts from agent 1 and synthesizes diagnosis."""

    def __init__(self, name="agent-two"):
        super().__init__(
            name=name,
            role=AgentRole.DIAGNOSTICIAN,
            description="Diagnoses based on observed facts",
            capabilities=[AgentCapability.INCIDENT_DIAGNOSIS, AgentCapability.SYNTHESIS],
        )

    def plan(self, task: AgentTask, context: AgentContext) -> List[str]:
        return ["Evaluate facts from prior agents"]

    def step(self, task: AgentTask, context: AgentContext, step_num: int) -> AgentStep:
        # Verify it received facts from FirstPipelineAgent
        prior_facts = context.facts
        has_errors = any("500" in f for f in prior_facts)
        return AgentStep(
            step_number=step_num,
            thought=f"Evaluating facts (found prior error facts: {has_errors})",
            action="COMPLETE",
            observation="Correlated to backend service failure",
        )

    def synthesize(self, task: AgentTask, context: AgentContext) -> AgentResult:
        prior_findings = context.variables.get("agent-one_findings", [])
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            summary=f"Diagnosed backend failure based on {len(prior_findings)} upstream findings",
            findings=[{"diagnosis": "Backend service degraded"}],
            duration_seconds=0.05,
        )


class FailingAgent(BaseAgent):
    """Agent that fails on execution to test pipeline error handling."""

    def __init__(self, name="failing-agent"):
        super().__init__(
            name=name,
            role=AgentRole.OBSERVER,
            description="Agent designed to fail",
            capabilities=[AgentCapability.READ_TELEMETRY],
        )

    def plan(self, task: AgentTask, context: AgentContext) -> List[str]:
        return ["Attempt operation"]

    def step(self, task: AgentTask, context: AgentContext, step_num: int) -> AgentStep:
        raise ValueError("Simulated hardware error")

    def synthesize(self, task: AgentTask, context: AgentContext) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_name=self.name,
            status=AgentStatus.FAILED,
            summary="Failed",
            duration_seconds=0.01,
        )


def test_orchestrator_agent_registration_and_lookup():
    """Verify registering, retrieving, and unregistering agents."""
    orch = AgentOrchestrator()
    agent = FirstPipelineAgent(name="log-observer")

    orch.register_agent(agent)
    assert len(orch.list_agents()) == 1

    # Lookup by name (case-insensitive)
    found = orch.get_agent("log-observer")
    assert found.agent_id == agent.agent_id
    assert found.name == "log-observer"

    # Lookup by agent_id
    found_by_id = orch.get_agent(agent.agent_id)
    assert found_by_id.name == "log-observer"

    # Capability filter
    log_agents = orch.find_agents_by_capability(AgentCapability.ANALYZE_LOGS)
    assert len(log_agents) == 1
    diag_agents = orch.find_agents_by_capability(AgentCapability.INCIDENT_DIAGNOSIS)
    assert len(diag_agents) == 0

    # Unregister
    assert orch.unregister_agent("log-observer") is True
    assert len(orch.list_agents()) == 0
    with pytest.raises(AgentNotFoundError):
        orch.get_agent("log-observer")


def test_orchestrator_dispatch_single_agent():
    """Verify task dispatch to designated agent."""
    orch = AgentOrchestrator()
    agent = FirstPipelineAgent(name="scanner")
    orch.register_agent(agent)

    task = AgentTask(description="Scan service logs for errors")
    result = orch.dispatch(task, agent_name="scanner")

    assert result.status == AgentStatus.COMPLETED
    assert result.agent_name == "scanner"
    assert "Found 5 HTTP 500 errors" in result.summary


def test_orchestrator_dispatch_default_agent():
    """Verify dispatching without agent_name selects registered agent."""
    orch = AgentOrchestrator()
    agent = FirstPipelineAgent(name="default-agent")
    orch.register_agent(agent)

    task = AgentTask(description="Inspect default telemetry")
    result = orch.dispatch(task)

    assert result.status == AgentStatus.COMPLETED
    assert result.agent_name == "default-agent"


def test_orchestrator_dispatch_empty_registry_raises():
    """Verify dispatch on empty orchestrator raises AgentNotFoundError."""
    orch = AgentOrchestrator()
    task = AgentTask(description="Any task")
    with pytest.raises(AgentNotFoundError):
        orch.dispatch(task)


def test_orchestrator_sequential_pipeline():
    """Verify sequential pipeline chains context facts and variables across agents."""
    orch = AgentOrchestrator()
    a1 = FirstPipelineAgent(name="agent-one")
    a2 = SecondPipelineAgent(name="agent-two")
    orch.register_agent(a1)
    orch.register_agent(a2)

    task = AgentTask(description="Run end-to-end diagnostic pipeline")
    results = orch.run_pipeline(task, agent_names=["agent-one", "agent-two"])

    assert len(results) == 2
    assert results[0].status == AgentStatus.COMPLETED
    assert results[0].agent_name == "agent-one"
    assert results[1].status == AgentStatus.COMPLETED
    assert results[1].agent_name == "agent-two"
    assert "Diagnosed backend failure based on 1 upstream findings" in results[1].summary


def test_orchestrator_pipeline_halts_on_failure():
    """Verify pipeline halts early if an intermediate agent fails."""
    orch = AgentOrchestrator()
    a1 = FailingAgent(name="bad-agent")
    a2 = SecondPipelineAgent(name="never-reached")
    orch.register_agent(a1)
    orch.register_agent(a2)

    task = AgentTask(description="Faulty pipeline task")
    results = orch.run_pipeline(task, agent_names=["bad-agent", "never-reached"])

    # Should only run bad-agent and stop
    assert len(results) == 1
    assert results[0].status == AgentStatus.FAILED
    assert "Simulated hardware error" in results[0].error


def test_orchestrator_broadcast():
    """Verify broadcasting task to multiple independent agents."""
    orch = AgentOrchestrator()
    a1 = FirstPipelineAgent(name="worker-1")
    a2 = FirstPipelineAgent(name="worker-2")
    orch.register_agent(a1)
    orch.register_agent(a2)

    task = AgentTask(description="Broadcast query")
    results = orch.broadcast(task, agent_names=["worker-1", "worker-2"])

    assert len(results) == 2
    assert results["worker-1"].status == AgentStatus.COMPLETED
    assert results["worker-2"].status == AgentStatus.COMPLETED


def test_orchestrator_reset_all():
    """Verify reset_all resets all agents to IDLE state."""
    orch = AgentOrchestrator()
    agent = FirstPipelineAgent(name="resettable")
    orch.register_agent(agent)

    task = AgentTask(description="Run task")
    orch.dispatch(task)
    assert agent.status == AgentStatus.COMPLETED

    orch.reset_all()
    assert agent.status == AgentStatus.IDLE
