# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Data models for AI agent execution, context state, and task orchestration."""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class AgentRole(str, Enum):
    """Standardized roles defining agent purpose and operational focus."""

    DIAGNOSTICIAN = "DIAGNOSTICIAN"
    OBSERVER = "OBSERVER"
    ADVISOR = "ADVISOR"
    COORDINATOR = "COORDINATOR"
    GENERAL = "GENERAL"


class AgentStatus(str, Enum):
    """Lifecycle and operational state of an AI agent."""

    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AgentCapability(str, Enum):
    """Discrete capabilities supported by agent implementations."""

    READ_TELEMETRY = "READ_TELEMETRY"
    ANALYZE_LOGS = "ANALYZE_LOGS"
    INSPECT_WORKLOADS = "INSPECT_WORKLOADS"
    INCIDENT_DIAGNOSIS = "INCIDENT_DIAGNOSIS"
    REMEDIATION_ADVISORY = "REMEDIATION_ADVISORY"
    SYNTHESIS = "SYNTHESIS"


class AgentLimits(BaseModel):
    """Configurable execution limits and safety boundaries for agent operations."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum iterations in reasoning/action loop",
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=0.01,
        le=600.0,
        description="Maximum total execution timeout in seconds",
    )
    max_input_chars: int = Field(
        default=16000,
        ge=100,
        le=100000,
        description="Maximum characters allowed for untrusted context input",
    )
    read_only: bool = Field(
        default=True,
        description="Enforces strict read-only mode (mutating actions forbidden)",
    )


class AgentStep(BaseModel):
    """Discrete reasoning step or action performed during task execution."""

    model_config = ConfigDict(extra="forbid")

    step_number: int = Field(ge=1, description="1-based step counter")
    thought: str = Field(description="Internal reasoning or planning step")
    action: str = Field(description="Action executed or diagnostic inquiry evaluated")
    observation: Optional[str] = Field(
        default=None, description="Observation or result from the action"
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


class AgentTask(BaseModel):
    """Instruction or diagnostic goal assigned to an agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        default_factory=lambda: f"TASK-{uuid.uuid4().hex[:8].upper()}",
        description="Unique task identifier",
    )
    description: str = Field(min_length=1, description="Goal or task description")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual input data (telemetry, alerts, logs, metrics)",
    )
    read_only: bool = Field(
        default=True,
        description="Mandates read-only execution; mutating commands strictly forbidden",
    )
    timeout: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=600.0,
        description="Optional task-specific timeout override",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Comprehensive execution result produced by an agent upon task completion."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent_id: str
    agent_name: str
    status: AgentStatus
    summary: str
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    duration_seconds: float = Field(ge=0.0)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    """Shared execution context, facts, and session state across agent iterations."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        default_factory=lambda: f"SESS-{uuid.uuid4().hex[:8].upper()}",
        description="Unique session identifier for multi-agent traces",
    )
    facts: List[str] = Field(
        default_factory=list,
        description="Accumulated verified facts or observations",
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scoped variables or state passed between steps/agents",
    )
    history: List[AgentStep] = Field(
        default_factory=list,
        description="Chronological record of executed steps in this session",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
