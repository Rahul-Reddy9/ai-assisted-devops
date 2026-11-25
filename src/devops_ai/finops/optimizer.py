# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Resource utilization analyzer and rightsizing optimization engine."""

from typing import Dict, List, Optional

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.core.logging import get_logger
from devops_ai.finops.models import (
    CloudProvider,
    CloudResource,
    OptimizationAction,
    OptimizationRecommendation,
    OptimizationRisk,
    OptimizationSummary,
    ResourceCategory,
    ResourceUtilization,
)

logger = get_logger("finops.optimizer")


class ResourceOptimizer:
    """Analyzes cloud infrastructure inventory and utilization to generate actionable savings recommendations."""

    def __init__(
        self,
        idle_cpu_threshold: float = 2.0,
        underutilized_cpu_threshold: float = 15.0,
        idle_memory_threshold: float = 15.0,
        min_observation_points: int = 1,
    ) -> None:
        """Initialize optimizer with configurable utilization thresholds."""
        self.idle_cpu_threshold = idle_cpu_threshold
        self.underutilized_cpu_threshold = underutilized_cpu_threshold
        self.idle_memory_threshold = idle_memory_threshold
        self.min_observation_points = min_observation_points

    def analyze_resource(
        self,
        resource: CloudResource,
        telemetry: Optional[List[ResourceUtilization]] = None,
    ) -> Optional[OptimizationRecommendation]:
        """Evaluate a single cloud resource against rightsizing and orphan scenarios."""
        if not telemetry or len(telemetry) < self.min_observation_points:
            logger.debug(
                f"Skipping resource {resource.resource_id}: insufficient telemetry points "
                f"({len(telemetry) if telemetry else 0} < {self.min_observation_points})"
            )
            return None

        # Extract metric series
        cpu_avg_vals = [p.avg_cpu_pct for p in telemetry if p.avg_cpu_pct is not None]
        cpu_peak_vals = [p.peak_cpu_pct for p in telemetry if p.peak_cpu_pct is not None]
        mem_avg_vals = [p.avg_memory_pct for p in telemetry if p.avg_memory_pct is not None]
        net_in_vals = [p.network_in_bytes for p in telemetry if p.network_in_bytes is not None]
        net_out_vals = [p.network_out_bytes for p in telemetry if p.network_out_bytes is not None]
        disk_read_vals = [
            p.disk_read_bytes_per_sec for p in telemetry if p.disk_read_bytes_per_sec is not None
        ]
        disk_write_vals = [
            p.disk_write_bytes_per_sec for p in telemetry if p.disk_write_bytes_per_sec is not None
        ]
        attached_states = [p.is_attached for p in telemetry if p.is_attached is not None]

        mean_cpu = sum(cpu_avg_vals) / len(cpu_avg_vals) if cpu_avg_vals else None
        max_peak_cpu = max(cpu_peak_vals) if cpu_peak_vals else None
        mean_mem = sum(mem_avg_vals) / len(mem_avg_vals) if mem_avg_vals else None
        total_net_bytes = sum(net_in_vals) + sum(net_out_vals)
        total_disk_io = sum(disk_read_vals) + sum(disk_write_vals)
        is_attached = attached_states[-1] if attached_states else None

        # -------------------------------------------------------------
        # Scenario 1: Idle Compute Instances (TERMINATE)
        # -------------------------------------------------------------
        if resource.category == ResourceCategory.COMPUTE:
            if (
                mean_cpu is not None
                and mean_cpu <= self.idle_cpu_threshold
                and (max_peak_cpu is None or max_peak_cpu <= 5.0)
                and total_net_bytes < 100_000.0  # < 100 KB total network
            ):
                confidence = min(0.98, 0.85 + 0.01 * len(telemetry))

                # Analyze pricing model and commitment coverage
                pricing_model = (
                    str(
                        resource.specs.get("pricing_model")
                        or resource.specs.get("term")
                        or resource.tags.get("pricing_model")
                        or ""
                    )
                    .strip()
                    .lower()
                )

                has_commitment = (
                    resource.specs.get("has_active_commitment") is True
                    or resource.specs.get("is_reserved") is True
                    or pricing_model in ("reserved", "savings_plan", "cud", "committed")
                    or resource.tags.get("commitment") in ("reserved", "savings_plan", "cud")
                )

                # Explicitly verified uncommitted on-demand or spot instance with confirmed zero commitment coverage
                is_verified_uncommitted = (
                    resource.specs.get("has_active_commitment") is False
                    or resource.specs.get("commitment_covered") is False
                    or resource.specs.get("commitment_status") in ("none", "uncommitted")
                ) and pricing_model in ("on-demand", "ondemand", "spot")

                evidence = [
                    f"Average CPU utilization was {mean_cpu:.1f}% (idle threshold <= {self.idle_cpu_threshold:.1f}%) across {len(telemetry)} observation periods.",
                    f"Peak CPU utilization did not exceed {max_peak_cpu or 0.0:.1f}%.",
                    f"Negligible network traffic recorded ({total_net_bytes / 1024.0:.1f} KB total).",
                    f"Direct resource-level billing elimination: ${resource.monthly_cost:.2f}/mo ({resource.currency}).",
                ]

                if has_commitment:
                    is_guaranteed = False
                    evidence.append(
                        f"Instance is associated with a commitment ({pricing_model or 'Reserved/Savings Plan'}); terminating the instance will not eliminate commitment charges or reduce the net invoice unless absorbed by another workload."
                    )
                elif is_verified_uncommitted:
                    is_guaranteed = True
                    evidence.append(
                        "Instance is confirmed on-demand/spot with no active commitment coverage; termination provides direct guaranteed net invoice savings."
                    )
                else:
                    is_guaranteed = False
                    evidence.append(
                        "Commitment coverage information is unavailable. Net invoice savings are estimated; terminating this instance eliminates direct on-demand resource charges but may not reduce costs if covered by active Reserved Instances (RIs) or Savings Plans."
                    )

                return OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    service_name=resource.service_name,
                    action=OptimizationAction.TERMINATE,
                    risk=OptimizationRisk.LOW
                    if resource.monthly_cost < 300.0
                    else OptimizationRisk.MEDIUM,
                    current_cost_monthly=round(resource.monthly_cost, 2),
                    projected_cost_monthly=0.0,
                    estimated_monthly_savings=round(resource.monthly_cost, 2),
                    confidence_score=round(confidence, 2),
                    evidence=evidence,
                    currency=resource.currency,
                    is_guaranteed=is_guaranteed,
                )

        # -------------------------------------------------------------
        # Scenario 2: Underutilized Compute Instances (DOWNSIZE)
        # -------------------------------------------------------------
        if resource.category == ResourceCategory.COMPUTE:
            if (
                mean_cpu is not None
                and mean_cpu <= self.underutilized_cpu_threshold
                and (max_peak_cpu is None or max_peak_cpu <= 40.0)
                and (mean_cpu > self.idle_cpu_threshold or total_net_bytes >= 100_000.0)
            ):
                savings = round(resource.monthly_cost * 0.50, 2)
                confidence = min(0.90, 0.75 + 0.01 * len(telemetry))
                evidence = [
                    f"Average CPU utilization was {mean_cpu:.1f}% (underutilized threshold <= {self.underutilized_cpu_threshold:.1f}%) across {len(telemetry)} observation periods.",
                    f"Peak CPU reached {max_peak_cpu or 0.0:.1f}%, leaving ample operational headroom (> 60%).",
                    "Projected 50% monthly cost savings by downsizing to an adjacent instance profile.",
                ]
                current_type = resource.specs.get("instance_type") or resource.specs.get(
                    "machine_type"
                )
                rec_spec = (
                    f"Downsize {current_type} to 1-tier lower profile"
                    if current_type
                    else "Downsize 1 instance tier"
                )
                return OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    service_name=resource.service_name,
                    action=OptimizationAction.DOWNSIZE,
                    risk=OptimizationRisk.LOW
                    if (max_peak_cpu and max_peak_cpu < 25.0)
                    else OptimizationRisk.MEDIUM,
                    current_cost_monthly=round(resource.monthly_cost, 2),
                    projected_cost_monthly=round(resource.monthly_cost - savings, 2),
                    estimated_monthly_savings=savings,
                    confidence_score=round(confidence, 2),
                    evidence=evidence,
                    recommended_specs=rec_spec,
                    currency=resource.currency,
                    is_guaranteed=False,
                )

        # -------------------------------------------------------------
        # Scenario 3: Unattached Storage Volumes (CLEANUP_ORPHAN)
        # -------------------------------------------------------------
        if resource.category == ResourceCategory.STORAGE:
            if is_attached is False and total_disk_io == 0.0:
                evidence = [
                    "Storage volume is unattached to any compute instance or virtual machine.",
                    "Zero disk read and write I/O operations recorded throughout the observation window.",
                    "Recommend taking an archive snapshot and terminating unattached orphaned volume.",
                ]
                return OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    service_name=resource.service_name,
                    action=OptimizationAction.CLEANUP_ORPHAN,
                    risk=OptimizationRisk.LOW,
                    current_cost_monthly=round(resource.monthly_cost, 2),
                    projected_cost_monthly=0.0,
                    estimated_monthly_savings=round(resource.monthly_cost, 2),
                    confidence_score=0.95,
                    evidence=evidence,
                    currency=resource.currency,
                    is_guaranteed=True,
                )

        # -------------------------------------------------------------
        # Scenario 4: Over-Provisioned Databases (DOWNSIZE)
        # -------------------------------------------------------------
        if resource.category == ResourceCategory.DATABASE:
            if (
                mean_cpu is not None
                and mean_cpu <= 10.0
                and (mean_mem is None or mean_mem <= self.idle_memory_threshold)
            ):
                savings = round(resource.monthly_cost * 0.40, 2)
                confidence = min(0.88, 0.75 + 0.01 * len(telemetry))
                evidence = [
                    f"Database average CPU utilization was {mean_cpu:.1f}% (threshold <= 10.0%) across {len(telemetry)} observation periods.",
                ]
                if mean_mem is not None:
                    evidence.append(
                        f"Database average memory utilization was {mean_mem:.1f}% (threshold <= {self.idle_memory_threshold:.1f}%)."
                    )
                evidence.append(
                    "Database can be safely rightsized to an adjacent tier during a scheduled maintenance window."
                )
                current_class = resource.specs.get("instance_class") or resource.specs.get("tier")
                rec_spec = (
                    f"Downsize {current_class} to lower compute tier"
                    if current_class
                    else "Downsize compute tier"
                )
                return OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    service_name=resource.service_name,
                    action=OptimizationAction.DOWNSIZE,
                    risk=OptimizationRisk.MEDIUM,  # Database operations involve maintenance windows
                    current_cost_monthly=round(resource.monthly_cost, 2),
                    projected_cost_monthly=round(resource.monthly_cost - savings, 2),
                    estimated_monthly_savings=savings,
                    confidence_score=round(confidence, 2),
                    evidence=evidence,
                    recommended_specs=rec_spec,
                    currency=resource.currency,
                    is_guaranteed=False,
                )

        # -------------------------------------------------------------
        # Scenario 5: Orphaned Static Public IP Addresses (CLEANUP_ORPHAN)
        # -------------------------------------------------------------
        is_ip_resource = (
            resource.category == ResourceCategory.NETWORK
            or "ip" in resource.service_name.lower()
            or "public_ip" in resource.specs
            or "allocation_id" in resource.specs
        )
        if is_ip_resource:
            if is_attached is False and total_net_bytes == 0.0:
                evidence = [
                    "Static public IP address is unassociated with any active network interface or compute workload.",
                    "Cloud providers assess hourly idle reservation charges for unassociated public IPv4 addresses.",
                    "Release orphaned IP address back to the provider pool.",
                ]
                return OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    service_name=resource.service_name,
                    action=OptimizationAction.CLEANUP_ORPHAN,
                    risk=OptimizationRisk.LOW,
                    current_cost_monthly=round(resource.monthly_cost, 2),
                    projected_cost_monthly=0.0,
                    estimated_monthly_savings=round(resource.monthly_cost, 2),
                    confidence_score=0.98,
                    evidence=evidence,
                    currency=resource.currency,
                    is_guaranteed=True,
                )

        return None

    def optimize_inventory(
        self,
        resources: List[CloudResource],
        utilization_map: Dict[str, List[ResourceUtilization]],
        base_currency: Optional[str] = None,
        provider_filter: Optional[CloudProvider] = None,
        category_filter: Optional[ResourceCategory] = None,
        region_filter: Optional[str] = None,
        min_savings: float = 0.0,
        risk_limit: Optional[OptimizationRisk] = None,
    ) -> OptimizationSummary:
        """Run optimization analysis across the entire cloud resource inventory."""
        if not resources:
            return OptimizationSummary(
                total_monthly_savings=0.0,
                total_annual_savings=0.0,
                guaranteed_monthly_savings=0.0,
                estimated_monthly_savings=0.0,
                currency=base_currency or "USD",
                recommendations_count=0,
                recommendations_by_action={},
                recommendations_by_risk={},
                recommendations=[],
            )

        # Enforce currency consistency
        target_currency = (base_currency or resources[0].currency).strip().upper()
        for idx, res in enumerate(resources):
            r_curr = res.currency.strip().upper()
            if r_curr != target_currency:
                logger.error(
                    f"Resource {idx} ({res.resource_id}) currency '{r_curr}' does not match expected '{target_currency}'"
                )
                raise CurrencyMismatchError(base_currency=target_currency, found_currency=r_curr)

        # Apply dimensional filters
        filtered_resources = resources
        if provider_filter:
            filtered_resources = [r for r in filtered_resources if r.provider == provider_filter]
        if category_filter:
            filtered_resources = [r for r in filtered_resources if r.category == category_filter]
        if region_filter:
            reg_clean = region_filter.strip().lower()
            filtered_resources = [
                r for r in filtered_resources if r.region and reg_clean in r.region.lower()
            ]

        # Risk priority mapping
        risk_priority = {
            OptimizationRisk.LOW: 1,
            OptimizationRisk.MEDIUM: 2,
            OptimizationRisk.HIGH: 3,
        }
        max_risk_level = risk_priority[risk_limit] if risk_limit else 3

        recommendations: List[OptimizationRecommendation] = []
        total_monthly_savings = 0.0
        guaranteed_monthly_savings = 0.0
        estimated_monthly_savings = 0.0
        actions_count: Dict[str, int] = {}
        risks_count: Dict[str, int] = {}

        for res in filtered_resources:
            telemetry = utilization_map.get(res.resource_id, [])
            rec = self.analyze_resource(res, telemetry)
            if not rec:
                continue

            # Check min savings threshold
            if rec.estimated_monthly_savings < min_savings:
                continue

            # Check risk ceiling
            if risk_priority[rec.risk] > max_risk_level:
                continue

            recommendations.append(rec)
            total_monthly_savings += rec.estimated_monthly_savings
            if rec.is_guaranteed:
                guaranteed_monthly_savings += rec.estimated_monthly_savings
            else:
                estimated_monthly_savings += rec.estimated_monthly_savings

            actions_count[rec.action.value] = actions_count.get(rec.action.value, 0) + 1
            risks_count[rec.risk.value] = risks_count.get(rec.risk.value, 0) + 1

        # Sort recommendations by monthly savings descending
        recommendations.sort(key=lambda r: r.estimated_monthly_savings, reverse=True)

        return OptimizationSummary(
            total_monthly_savings=round(total_monthly_savings, 2),
            total_annual_savings=round(total_monthly_savings * 12.0, 2),
            guaranteed_monthly_savings=round(guaranteed_monthly_savings, 2),
            estimated_monthly_savings=round(estimated_monthly_savings, 2),
            currency=target_currency,
            recommendations_count=len(recommendations),
            recommendations_by_action=actions_count,
            recommendations_by_risk=risks_count,
            recommendations=recommendations,
        )
