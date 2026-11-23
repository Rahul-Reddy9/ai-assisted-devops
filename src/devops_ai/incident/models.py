# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Data models for incident detection, investigation, and lifecycle management."""

import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    """Standardized incident and alert severity levels."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Category(str, Enum):
    """Broad categorization of infrastructure and operational failures."""

    INFRASTRUCTURE = "INFRASTRUCTURE"
    CONTAINER = "CONTAINER"
    APPLICATION = "APPLICATION"
    NETWORK = "NETWORK"
    PROVIDER = "PROVIDER"
    UNKNOWN = "UNKNOWN"


class IncidentStatus(str, Enum):
    """Finite state machine states for incident lifecycle management."""

    DETECTED = "DETECTED"
    CORRELATED = "CORRELATED"
    INVESTIGATING = "INVESTIGATING"
    ANALYZED = "ANALYZED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class AlertSource(str, Enum):
    """Source subsystem generating the operational alert."""

    LOGS = "LOGS"
    METRICS = "METRICS"
    HEALTH_PROBE = "HEALTH_PROBE"
    DOCKER = "DOCKER"
    KUBERNETES = "KUBERNETES"
    MANUAL = "MANUAL"


class RiskLevel(str, Enum):
    """Advisory risk categorization for remediation recommendations."""

    SAFE = "SAFE"
    MODERATE = "MODERATE"
    DANGEROUS = "DANGEROUS"


class Alert(BaseModel):
    """Discrete operational alert ingested from an observability source."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(description="Unique alert identifier e.g. ALT-20261123-0001")
    source: AlertSource
    timestamp: datetime.datetime
    component: str = Field(description="Affected component e.g. web-app, postgres, worker-pod")
    message: str
    severity: Severity = Severity.MEDIUM
    fingerprint: str = Field(description="SHA-256 hash of source + component + error pattern")
    recurrence_count: int = Field(default=1, ge=1)
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class TimelineEntry(BaseModel):
    """Chronological event entry in an incident timeline."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime.datetime
    source: AlertSource
    component: str
    event: str
    severity: Severity


class CorrelatedEvidence(BaseModel):
    """Evidence bundle containing raw observability snapshots correlated for an incident."""

    model_config = ConfigDict(extra="forbid")

    evidence_catalog: Dict[str, str] = Field(
        default_factory=dict,
        description="Catalog mapping evidence IDs e.g. EVID-LOG-01 to raw text/values",
    )
    log_snippets: List[str] = Field(default_factory=list)
    metric_anomalies: Dict[str, float] = Field(default_factory=dict)
    container_states: List[Dict[str, Any]] = Field(default_factory=list)
    k8s_events: List[Dict[str, Any]] = Field(default_factory=list)
    probe_failures: List[Dict[str, Any]] = Field(default_factory=list)


class RootCauseHypothesis(BaseModel):
    """Proposed root cause explanation with confidence and supporting evidence references."""

    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    tier: Literal["tier_1_deterministic", "tier_2_ai"]
    evidence_refs: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    is_confirmed: bool = False
    is_unsupported: bool = False


class RemediationAction(BaseModel):
    """Actionable, risk-categorized advisory recommendation with human verification steps."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    title: str
    description: str
    action_type: Literal["immediate_mitigation", "permanent_fix", "diagnostic"]
    risk_level: RiskLevel
    command: Optional[str] = None
    verification_step: str
    advisory_warning: str = (
        "ADVISORY ONLY: Must be executed manually by a human operator. "
        "Automated execution is strictly disabled."
    )


class Incident(BaseModel):
    """Complete record of an operational incident throughout its lifecycle."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(description="Unique incident identifier e.g. INC-20261123-A1B2")
    version: int = Field(
        default=1, ge=1, description="Monotonic version for optimistic concurrency"
    )
    title: str
    category: Category = Category.UNKNOWN
    severity: Severity = Severity.MEDIUM
    status: IncidentStatus = IncidentStatus.DETECTED
    created_at: datetime.datetime
    updated_at: datetime.datetime
    resolved_at: Optional[datetime.datetime] = None
    closed_at: Optional[datetime.datetime] = None
    resolution_notes: Optional[str] = None
    components_affected: List[str] = Field(default_factory=list)
    alerts: List[Alert] = Field(default_factory=list)
    timeline: List[TimelineEntry] = Field(default_factory=list)
    evidence: CorrelatedEvidence = Field(default_factory=CorrelatedEvidence)
    root_cause: Optional[RootCauseHypothesis] = None
    alternative_hypotheses: List[RootCauseHypothesis] = Field(default_factory=list)
    recommendations: List[RemediationAction] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
