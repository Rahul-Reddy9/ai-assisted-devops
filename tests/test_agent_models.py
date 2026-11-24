# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying AI agent data models, serialization, and constraints."""

import pytest
from pydantic import ValidationError

from devops_ai.agents.models import (
    AgentContext,
    AgentLimits,
    AgentResult,
    AgentRole,
    AgentStatus,
    AgentStep,
    AgentTask,
)


def test_agent_role_and_status_enums():
    """Verify standard agent roles and status state transitions."""
    assert AgentRole.DIAGNOSTICIAN == "DIAGNOSTICIAN"
    assert AgentRole.OBSERVER == "OBSERVER"
    assert AgentRole.ADVISOR == "ADVISOR"
    assert AgentRole.COORDINATOR == "COORDINATOR"
    assert AgentRole.GENERAL == "GENERAL"

    assert AgentStatus.IDLE == "IDLE"
    assert AgentStatus.INITIALIZING == "INITIALIZING"
    assert AgentStatus.PLANNING == "PLANNING"
    assert AgentStatus.EXECUTING == "EXECUTING"
    assert AgentStatus.COMPLETED == "COMPLETED"
    assert AgentStatus.FAILED == "FAILED"
    assert AgentStatus.CANCELLED == "CANCELLED"


def test_agent_limits_defaults_and_validation():
    """Verify AgentLimits validation and default safety boundaries."""
    limits = AgentLimits()
    assert limits.read_only is True
    assert limits.max_iterations == 10
    assert limits.timeout_seconds == 60.0
    assert limits.max_input_chars == 16000

    # Test out of bounds validation
    with pytest.raises(ValidationError):
        AgentLimits(max_iterations=0)

    with pytest.raises(ValidationError):
        AgentLimits(max_iterations=100)

    with pytest.raises(ValidationError):
        AgentLimits(timeout_seconds=0.0)

    with pytest.raises(ValidationError):
        AgentLimits(timeout_seconds=1000.0)


def test_agent_task_creation_and_defaults():
    """Verify AgentTask default initialization and field validation."""
    task = AgentTask(description="Diagnose latency spike on checkout-service")
    assert task.task_id.startswith("TASK-")
    assert task.read_only is True
    assert task.context == {}
    assert task.timeout is None

    # Empty description is rejected
    with pytest.raises(ValidationError):
        AgentTask(description="")


def test_agent_step_and_result_roundtrip():
    """Verify AgentStep and AgentResult serialization and round-trip fidelity."""
    step = AgentStep(
        step_number=1,
        thought="Check container restart reasons",
        action="INSPECT_CONTAINER_STATE",
        observation="Exit code 137 found in last termination status",
    )
    assert step.step_number == 1
    assert step.observation is not None

    result = AgentResult(
        task_id="TASK-TEST-001",
        agent_id="AGT-DIAG-01",
        agent_name="test-diagnostician",
        status=AgentStatus.COMPLETED,
        summary="Container experienced memory exhaustion (OOMKill).",
        findings=[{"type": "OOM", "confidence": 0.95}],
        steps=[step],
        duration_seconds=1.234,
        metadata={"read_only": True},
    )

    dumped = result.model_dump(mode="json")
    roundtrip = AgentResult.model_validate(dumped)

    assert roundtrip.task_id == "TASK-TEST-001"
    assert roundtrip.status == AgentStatus.COMPLETED
    assert len(roundtrip.steps) == 1
    assert roundtrip.steps[0].action == "INSPECT_CONTAINER_STATE"


def test_agent_context_accumulation():
    """Verify AgentContext tracks facts, variables, and history."""
    ctx = AgentContext()
    assert ctx.session_id.startswith("SESS-")
    assert len(ctx.facts) == 0

    ctx.facts.append("Database connection pool size is 20")
    ctx.variables["active_connections"] = 20
    ctx.history.append(
        AgentStep(
            step_number=1,
            thought="Inspect active connections",
            action="QUERY_METRICS",
            observation="Active connections: 20",
        )
    )

    assert len(ctx.facts) == 1
    assert ctx.variables["active_connections"] == 20
    assert len(ctx.history) == 1


def test_extra_forbidden_on_all_agent_models():
    """Verify that unexpected attributes are strictly forbidden across agent models."""
    with pytest.raises(ValidationError):
        AgentLimits(extra_attribute="not_allowed")  # type: ignore

    with pytest.raises(ValidationError):
        AgentTask(description="test", invalid_field=123)  # type: ignore

    with pytest.raises(ValidationError):
        AgentContext(unknown_key="fail")  # type: ignore
