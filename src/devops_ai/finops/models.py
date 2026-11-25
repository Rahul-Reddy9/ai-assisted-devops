# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Provider-independent cloud cost, resource, utilization, and FinOps data models."""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class CloudProvider(str, Enum):
    """Supported cloud infrastructure and service providers."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    CUSTOM = "custom"


class ResourceCategory(str, Enum):
    """Standardized classification of cloud resources."""

    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORK = "network"
    SERVERLESS = "serverless"
    OTHER = "other"


class BillingPeriod(BaseModel):
    """Temporal range defining an accounting or billing cycle."""

    model_config = ConfigDict(extra="forbid")

    start_time: datetime.datetime = Field(description="Start timestamp in UTC")
    end_time: datetime.datetime = Field(description="End timestamp in UTC")

    @property
    def duration_days(self) -> float:
        """Calculate the duration of the billing period in days."""
        delta = self.end_time - self.start_time
        return max(0.0, delta.total_seconds() / 86400.0)


class BillingRecord(BaseModel):
    """Normalized line item representing cloud expenditure and usage."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(
        default_factory=lambda: f"REC-{uuid.uuid4().hex[:10].upper()}",
        min_length=1,
        description="Unique identifier for the billing record",
    )
    provider: CloudProvider = Field(description="Cloud provider classification")
    account_id: str = Field(
        min_length=1, description="Account, subscription, or project identifier"
    )
    service_name: str = Field(min_length=1, description="Cloud service or product name")
    resource_id: Optional[str] = Field(
        default=None, description="Cloud resource ARN, URI, or identifier"
    )
    resource_name: Optional[str] = Field(
        default=None, description="Human-readable resource name or moniker"
    )
    region: Optional[str] = Field(
        default=None, description="Geographic cloud region (e.g. us-east-1, westeurope)"
    )
    usage_type: Optional[str] = Field(
        default=None, description="Usage category or billing line item type"
    )
    usage_quantity: float = Field(default=0.0, ge=0.0, description="Measured usage quantity")
    usage_unit: Optional[str] = Field(
        default=None, description="Unit of measurement (e.g. Hrs, GB-Mo, Requests)"
    )
    cost: float = Field(
        description="Direct monetary cost for this record (positive for charges, negative for credits/refunds)"
    )
    currency: str = Field(
        default="USD", min_length=3, max_length=3, description="ISO 4217 3-letter currency code"
    )
    start_time: datetime.datetime = Field(description="Period start time in UTC")
    end_time: datetime.datetime = Field(description="Period end time in UTC")
    tags: Dict[str, str] = Field(
        default_factory=dict, description="Metadata tags attached to the resource"
    )
    is_estimated: bool = Field(
        default=False, description="Whether this record is an estimate or verified actual invoice"
    )

    @property
    def is_credit(self) -> bool:
        """Indicate whether this record represents a credit, refund, or negative adjustment."""
        return self.cost < 0.0


class CloudResource(BaseModel):
    """Provider-independent representation of a cloud asset."""

    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(min_length=1, description="Global unique identifier of the resource")
    resource_name: Optional[str] = Field(default=None, description="Name or friendly label")
    provider: CloudProvider = Field(description="Cloud provider hosting the asset")
    account_id: str = Field(min_length=1, description="Parent account or project ID")
    service_name: str = Field(min_length=1, description="Associated cloud service")
    category: ResourceCategory = Field(description="Functional category")
    region: Optional[str] = Field(default=None, description="Cloud region")
    monthly_cost: float = Field(default=0.0, ge=0.0, description="Normalized monthly cost")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    specs: Dict[str, Any] = Field(
        default_factory=dict, description="Hardware or service specifications"
    )
    tags: Dict[str, str] = Field(default_factory=dict, description="Assigned resource tags")


class ResourceUtilization(BaseModel):
    """Historical or sampled utilization telemetry for a cloud asset."""

    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(min_length=1, description="Identifier of the target resource")
    service_name: str = Field(description="Associated service name")
    timestamp: datetime.datetime = Field(description="Timestamp of observation in UTC")
    avg_cpu_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Average CPU utilization percentage"
    )
    peak_cpu_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Peak CPU utilization percentage"
    )
    avg_memory_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Average memory utilization percentage"
    )
    peak_memory_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Peak memory utilization percentage"
    )
    disk_read_bytes_per_sec: Optional[float] = Field(
        default=None, ge=0.0, description="Disk read throughput"
    )
    disk_write_bytes_per_sec: Optional[float] = Field(
        default=None, ge=0.0, description="Disk write throughput"
    )
    network_in_bytes: Optional[float] = Field(
        default=None, ge=0.0, description="Ingress network bytes"
    )
    network_out_bytes: Optional[float] = Field(
        default=None, ge=0.0, description="Egress network bytes"
    )
    is_attached: Optional[bool] = Field(
        default=None, description="Whether volume or IP is actively attached"
    )


class OptimizationAction(str, Enum):
    """Recommended optimization operations."""

    TERMINATE = "TERMINATE"
    DOWNSIZE = "DOWNSIZE"
    STORAGE_TIER = "STORAGE_TIER"
    RESERVED_INSTANCE = "RESERVED_INSTANCE"
    CLEANUP_ORPHAN = "CLEANUP_ORPHAN"


class OptimizationRisk(str, Enum):
    """Operational risk level of applying an optimization."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class OptimizationRecommendation(BaseModel):
    """Specific, evidence-backed FinOps rightsizing or savings action."""

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(
        default_factory=lambda: f"REC-OPT-{uuid.uuid4().hex[:8].upper()}",
        min_length=1,
    )
    resource_id: str = Field(min_length=1, description="Target resource identifier")
    service_name: str = Field(min_length=1, description="Cloud service name")
    action: OptimizationAction = Field(description="Recommended action type")
    risk: OptimizationRisk = Field(description="Operational risk rating")
    current_cost_monthly: float = Field(ge=0.0, description="Current monthly cost")
    projected_cost_monthly: float = Field(ge=0.0, description="Projected monthly cost after action")
    estimated_monthly_savings: float = Field(ge=0.0, description="Estimated monthly dollar savings")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score based on telemetry duration and quality"
    )
    evidence: List[str] = Field(
        min_length=1, description="Verifiable metrics and facts supporting recommendation"
    )
    recommended_specs: Optional[str] = Field(
        default=None, description="Proposed target specifications (e.g. t3.medium -> t3.small)"
    )


class AnomalySeverity(str, Enum):
    """Classification of spending anomaly severity."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CostAnomaly(BaseModel):
    """Detected unexpected spend variance or spike."""

    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(
        default_factory=lambda: f"ANOM-{uuid.uuid4().hex[:8].upper()}", min_length=1
    )
    detected_date: datetime.date = Field(description="Date anomaly was observed")
    service_name: Optional[str] = Field(default=None, description="Service causing the anomaly")
    actual_cost: float = Field(ge=0.0, description="Observed cost on anomaly date")
    expected_cost: float = Field(ge=0.0, description="Expected baseline cost")
    cost_delta: float = Field(description="Difference between actual and expected spend")
    percentage_increase: float = Field(description="Percentage variance from baseline")
    z_score: float = Field(description="Statistical Z-score (standard deviations from mean)")
    severity: AnomalySeverity = Field(description="Severity classification")
    driver_resources: List[str] = Field(
        default_factory=list, description="Specific resource IDs responsible for the spike"
    )
    explanation: str = Field(min_length=1, description="Human-readable root-cause explanation")


class ForecastPoint(BaseModel):
    """A single projected time point in a cloud expenditure forecast."""

    model_config = ConfigDict(extra="forbid")

    date: datetime.date = Field(description="Projected forecast date")
    projected_cost: float = Field(ge=0.0, description="Estimated spend amount")
    lower_bound: float = Field(ge=0.0, description="Lower confidence interval bound")
    upper_bound: float = Field(ge=0.0, description="Upper confidence interval bound")


class CostForecast(BaseModel):
    """Forward-looking projection of cloud costs."""

    model_config = ConfigDict(extra="forbid")

    forecast_id: str = Field(
        default_factory=lambda: f"FCST-{uuid.uuid4().hex[:8].upper()}", min_length=1
    )
    generated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    days_projected: int = Field(ge=1, description="Number of future days forecasted")
    method: str = Field(description="Forecasting model or algorithm utilized")
    total_projected_spend: float = Field(ge=0.0, description="Sum of projected spend over window")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    points: List[ForecastPoint] = Field(description="Daily forecast points")
    evaluation_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Backtesting accuracy metrics (MAE, MAPE, RMSE)"
    )


class CostSummary(BaseModel):
    """Aggregated financial summary across analyzed billing records."""

    model_config = ConfigDict(extra="forbid")

    total_cost: float = Field(
        description="Net total spend across all analyzed records (gross charges plus credit adjustments)"
    )
    gross_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Gross total spend excluding credit/refund adjustments",
    )
    credit_cost: float = Field(
        default=0.0,
        le=0.0,
        description="Total credit, refund, or discount adjustments (negative or zero)",
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)
    period_start: Optional[datetime.datetime] = Field(
        default=None, description="Earliest billing timestamp in analyzed records"
    )
    period_end: Optional[datetime.datetime] = Field(
        default=None, description="Latest billing timestamp in analyzed records"
    )
    total_records: int = Field(ge=0, description="Count of billing records processed")
    service_costs: Dict[str, float] = Field(
        default_factory=dict, description="Breakdown of costs by cloud service"
    )
    account_costs: Dict[str, float] = Field(
        default_factory=dict, description="Breakdown of costs by account/subscription/project"
    )
    region_costs: Dict[str, float] = Field(
        default_factory=dict, description="Breakdown of costs by cloud region"
    )
    daily_costs: Dict[str, float] = Field(
        default_factory=dict, description="Daily aggregated spend keyed by 'YYYY-MM-DD'"
    )
    monthly_costs: Dict[str, float] = Field(
        default_factory=dict, description="Monthly aggregated spend keyed by 'YYYY-MM'"
    )
    top_resources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top spending individual resources"
    )
