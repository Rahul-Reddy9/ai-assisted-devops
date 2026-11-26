# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Provider-independent cloud cost, resource, utilization, and FinOps data models."""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    currency: str = Field(
        default="USD", min_length=3, max_length=3, description="ISO 4217 3-letter currency code"
    )
    is_guaranteed: bool = Field(
        default=False,
        description="Whether savings are guaranteed (e.g. unattached orphan cleanup) vs estimated",
    )

    @property
    def estimated_annual_savings(self) -> float:
        """Calculate annualized savings from estimated monthly savings."""
        return round(self.estimated_monthly_savings * 12.0, 2)


class OptimizationSummary(BaseModel):
    """Aggregated summary of infrastructure rightsizing and cost optimization recommendations."""

    model_config = ConfigDict(extra="forbid")

    total_monthly_savings: float = Field(
        ge=0.0, description="Total potential monthly savings sum across all recommendations"
    )
    total_annual_savings: float = Field(
        ge=0.0, description="Total potential annual savings sum across all recommendations"
    )
    guaranteed_monthly_savings: float = Field(
        default=0.0,
        ge=0.0,
        description="Portion of monthly savings from guaranteed actions (e.g. orphan cleanup)",
    )
    estimated_monthly_savings: float = Field(
        default=0.0,
        ge=0.0,
        description="Portion of monthly savings from estimated rightsizing",
    )
    currency: str = Field(
        default="USD", min_length=3, max_length=3, description="ISO 4217 3-letter currency code"
    )
    recommendations_count: int = Field(
        ge=0, description="Total count of optimization recommendations"
    )
    recommendations_by_action: Dict[str, int] = Field(
        default_factory=dict, description="Count of recommendations grouped by action"
    )
    recommendations_by_risk: Dict[str, int] = Field(
        default_factory=dict, description="Count of recommendations grouped by risk rating"
    )
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list, description="Detailed list of actionable recommendations"
    )

    @property
    def guaranteed_annual_savings(self) -> float:
        """Calculate annualized guaranteed savings from guaranteed monthly savings."""
        return round(self.guaranteed_monthly_savings * 12.0, 2)

    @property
    def estimated_annual_savings(self) -> float:
        """Calculate annualized estimated savings from estimated monthly savings."""
        return round(self.estimated_monthly_savings * 12.0, 2)


class AnomalySeverity(str, Enum):
    """Classification of spending anomaly severity."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CostAnomaly(BaseModel):
    """Detected unexpected spend variance or statistical surge."""

    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(
        default_factory=lambda: f"ANOM-{uuid.uuid4().hex[:8].upper()}", min_length=1
    )
    detected_date: datetime.date = Field(description="Date anomaly was observed")
    provider: Optional[CloudProvider] = Field(
        default=None, description="Cloud provider where anomaly originated"
    )
    account_id: Optional[str] = Field(
        default=None, description="Account or subscription identifier"
    )
    service_name: Optional[str] = Field(default=None, description="Service causing the anomaly")
    region: Optional[str] = Field(default=None, description="Cloud region of anomaly")
    resource_id: Optional[str] = Field(
        default=None, description="Primary resource identifier if isolated"
    )
    actual_cost: float = Field(ge=0.0, description="Observed cost on anomaly date")
    expected_cost: float = Field(ge=0.0, description="Expected baseline cost")
    cost_delta: float = Field(description="Difference between actual and expected spend")
    percentage_increase: float = Field(description="Percentage variance from baseline")
    z_score: float = Field(description="Statistical Z-score (standard deviations from mean)")
    severity: AnomalySeverity = Field(description="Severity classification")
    confidence_score: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Confidence score based on baseline stability and sample size",
    )
    detection_method: str = Field(
        default="z_score",
        description="Detection algorithm (e.g. 'z_score', 'mad', 'rolling_threshold')",
    )
    historical_observation_count: int = Field(
        default=0,
        ge=0,
        description="Number of historical observation days used to calculate baseline",
    )
    currency: str = Field(
        default="USD", min_length=3, max_length=3, description="ISO 4217 3-letter currency code"
    )
    driver_resources: List[str] = Field(
        default_factory=list, description="Specific resource IDs responsible for the spike"
    )
    explanation: str = Field(min_length=1, description="Human-readable root-cause explanation")
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting telemetry metrics, baselines, and statistical facts",
    )


class AnomalySummary(BaseModel):
    """Aggregated collection and dimensional summary of detected spending anomalies."""

    model_config = ConfigDict(extra="forbid")

    total_anomalies: int = Field(ge=0, description="Total count of anomalies detected")
    total_excess_spend: float = Field(
        ge=0.0, description="Sum of positive cost deltas above baseline"
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)
    anomalies_by_severity: Dict[str, int] = Field(default_factory=dict)
    anomalies_by_service: Dict[str, int] = Field(default_factory=dict)
    period_start: Optional[datetime.date] = None
    period_end: Optional[datetime.date] = None
    anomalies: List[CostAnomaly] = Field(default_factory=list)


class ForecastMethod(str, Enum):
    """Supported expenditure forecasting algorithms."""

    LINEAR_REGRESSION = "linear_regression"
    WEIGHTED_MOVING_AVERAGE = "weighted_moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    HISTORICAL_AVERAGE = "historical_average"


class ForecastPoint(BaseModel):
    """A single projected time point in a cloud expenditure forecast with prediction intervals."""

    model_config = ConfigDict(extra="forbid")

    date: datetime.date = Field(description="Projected forecast date")
    projected_cost: float = Field(ge=0.0, description="Estimated spend amount (P50)")
    lower_bound: float = Field(ge=0.0, description="Lower prediction interval bound (P10)")
    upper_bound: float = Field(ge=0.0, description="Upper prediction interval bound (P90)")
    p10: Optional[float] = Field(
        default=None, ge=0.0, description="Optimistic expenditure estimate (10th percentile)"
    )
    p50: Optional[float] = Field(
        default=None, ge=0.0, description="Median expenditure estimate (50th percentile)"
    )
    p90: Optional[float] = Field(
        default=None, ge=0.0, description="Conservative expenditure estimate (90th percentile)"
    )

    @model_validator(mode="after")
    def sync_and_validate_quantiles(self) -> "ForecastPoint":
        """Ensure p10, p50, p90 are populated and satisfy 0 <= p10 <= p50 <= p90."""
        if self.p10 is None:
            self.p10 = self.lower_bound
        if self.p50 is None:
            self.p50 = self.projected_cost
        if self.p90 is None:
            self.p90 = self.upper_bound
        # Enforce non-negative ordering invariant: 0 <= p10 <= p50 <= p90
        if self.p10 > self.p50:
            self.p10 = self.p50
        if self.p90 < self.p50:
            self.p90 = self.p50
        return self


class CostForecast(BaseModel):
    """Forward-looking projection of cloud costs with uncertainty intervals."""

    model_config = ConfigDict(extra="forbid")

    forecast_id: str = Field(
        default_factory=lambda: f"FCST-{uuid.uuid4().hex[:8].upper()}", min_length=1
    )
    generated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    target_dimension: str = Field(
        default="total", description="Target dimension for forecast (e.g. 'total', 'service:EC2')"
    )
    days_projected: int = Field(ge=1, description="Number of future days forecasted")
    method: str = Field(description="Forecasting model or algorithm utilized")
    total_projected_spend: float = Field(
        ge=0.0, description="Sum of projected spend over window (P50)"
    )
    total_p10_spend: float = Field(
        default=0.0, ge=0.0, description="Sum of optimistic P10 spend over window"
    )
    total_p90_spend: float = Field(
        default=0.0, ge=0.0, description="Sum of conservative P90 spend over window"
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)
    confidence_level: float = Field(
        default=0.90, ge=0.50, le=0.99, description="Confidence level for prediction interval"
    )
    historical_data_points: int = Field(
        default=0, ge=0, description="Number of historical observation days used"
    )
    trend: str = Field(
        default="stable",
        description="Expenditure trajectory ('increasing', 'decreasing', 'stable')",
    )
    points: List[ForecastPoint] = Field(description="Daily forecast points")
    evaluation_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Backtesting accuracy metrics (MAE, MAPE, RMSE)"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Methodology notes, trend evidence, and assumptions"
    )
    insufficient_data: bool = Field(
        default=False, description="Flag indicating historical observations were below threshold"
    )
    message: Optional[str] = Field(
        default=None, description="Informational or qualification message"
    )


class ForecastSummary(BaseModel):
    """Aggregated executive summary of cloud expenditure projections."""

    model_config = ConfigDict(extra="forbid")

    forecast_id: str = Field(min_length=1)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    horizon_days: int = Field(ge=1)
    forecast_start_date: datetime.date
    forecast_end_date: datetime.date
    total_projected_cost: float = Field(ge=0.0)
    total_p10_cost: float = Field(ge=0.0)
    total_p90_cost: float = Field(ge=0.0)
    daily_average_cost: float = Field(ge=0.0)
    trend: str = Field(default="stable")
    method: str
    confidence_level: float
    historical_days_analyzed: int = Field(ge=0)
    forecast: CostForecast


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


class FinOpsAdvisoryRecommendation(BaseModel):
    """Actionable FinOps recommendation synthesizing cost, anomaly, rightsizing, or forecast insights."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(min_length=1, description="Brief summary title of the recommendation")
    category: str = Field(
        description="Classification: compute, storage, database, network, anomaly, general"
    )
    priority: str = Field(
        default="medium", description="Priority level: critical, high, medium, low"
    )
    savings_type: str = Field(
        default="estimated",
        description="Savings classification: guaranteed (idle/unattached cleanup) or estimated (rightsizing/architecture)",
    )
    potential_monthly_savings: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated or guaranteed monthly savings in specified currency",
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)
    confidence_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    action: str = Field(
        min_length=1,
        description="Specific actionable steps to capture savings or mitigate cost risk",
    )
    evidence: List[str] = Field(
        default_factory=list, description="Factual evidence citations supporting the recommendation"
    )
    affected_resources: List[str] = Field(
        default_factory=list, description="Cloud resource IDs associated with this recommendation"
    )

    @property
    def potential_annual_savings(self) -> float:
        """Calculate annualized cost savings."""
        return round(self.potential_monthly_savings * 12.0, 2)


class FinOpsAdvisoryReport(BaseModel):
    """Holistic executive FinOps advisory report synthesizing cost findings, risks, and strategic guidance."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="FinOps Strategic Advisory Report", min_length=1)
    generated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    executive_summary: str = Field(
        min_length=1,
        description="High-level narrative synthesizing spending posture, key cost drivers, and priority interventions",
    )
    recommendations: List[FinOpsAdvisoryRecommendation] = Field(
        default_factory=list,
        description="Prioritized list of actionable cost optimization recommendations",
    )
    total_monthly_savings: float = Field(
        default=0.0,
        ge=0.0,
        description="Total identified monthly savings across all recommendations",
    )
    guaranteed_monthly_savings: float = Field(
        default=0.0,
        ge=0.0,
        description="Immediate guaranteed monthly savings from zero-utilization/unattached cleanup",
    )
    estimated_monthly_savings: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated monthly savings from rightsizing and workload adjustments",
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)
    anomalies_detected: int = Field(
        default=0, ge=0, description="Number of cost anomalies accounted for"
    )
    historical_excess_spend: float = Field(
        default=0.0,
        ge=0.0,
        description="Historical anomalous expenditure identified across observation windows (cost avoidance risk, not recurring savings)",
    )
    forecast_trend: str = Field(
        default="stable",
        description="Underlying spending trajectory (increasing, decreasing, stable)",
    )
    provider: str = Field(
        default="offline", description="LLM provider or rule-based engine that generated the advice"
    )
    model: Optional[str] = Field(
        default=None, description="LLM model identifier used for synthesis, if applicable"
    )
    disclaimer: str = Field(
        default=(
            "This advisory report is generated for strategic planning purposes only. "
            "Automated cloud mutations are disabled. All recommendations must be reviewed "
            "and validated before applying changes in production environments."
        ),
        description="Standard safety and non-mutation disclaimer",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operational metadata, engine flags, and timing statistics",
    )

    @model_validator(mode="after")
    def compute_savings_if_unset(self) -> "FinOpsAdvisoryReport":
        """Compute aggregated savings totals if recommendations are provided and totals are zero."""
        if self.recommendations and self.total_monthly_savings == 0.0:
            guaranteed = sum(
                r.potential_monthly_savings
                for r in self.recommendations
                if r.savings_type == "guaranteed"
            )
            estimated = sum(
                r.potential_monthly_savings
                for r in self.recommendations
                if r.savings_type != "guaranteed"
            )
            self.guaranteed_monthly_savings = round(guaranteed, 2)
            self.estimated_monthly_savings = round(estimated, 2)
            self.total_monthly_savings = round(guaranteed + estimated, 2)
        return self

    @property
    def total_annual_savings(self) -> float:
        """Calculate annualized total cost savings."""
        return round(self.total_monthly_savings * 12.0, 2)

    @property
    def guaranteed_annual_savings(self) -> float:
        """Calculate annualized guaranteed cost savings."""
        return round(self.guaranteed_monthly_savings * 12.0, 2)

    @property
    def estimated_annual_savings(self) -> float:
        """Calculate annualized estimated cost savings."""
        return round(self.estimated_monthly_savings * 12.0, 2)
