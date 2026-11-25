# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit and scenario tests for FinOps resource utilization analysis and rightsizing optimizer."""

import datetime
import pytest

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.finops.importer import get_synthetic_resource_inventory
from devops_ai.finops.models import (
    CloudProvider,
    CloudResource,
    OptimizationAction,
    OptimizationRisk,
    ResourceCategory,
    ResourceUtilization,
)
from devops_ai.finops.optimizer import ResourceOptimizer


class TestResourceOptimizerScenarios:
    def setup_method(self):
        self.optimizer = ResourceOptimizer()
        self.now = datetime.datetime.now(datetime.timezone.utc)

    def test_idle_compute_scenario_default_unspecified_commitment(self):
        """Idle compute when commitment info is unavailable: classified as estimated net savings."""
        res = CloudResource(
            resource_id="i-test-idle-01",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.COMPUTE,
            region="us-east-1",
            monthly_cost=160.00,
            currency="USD",
            specs={"instance_type": "m5.large"},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                avg_cpu_pct=0.4,
                peak_cpu_pct=2.1,
                network_in_bytes=0.0,
                network_out_bytes=0.0,
                is_attached=True,
            )
            for i in range(10)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is not None
        assert rec.action == OptimizationAction.TERMINATE
        assert rec.estimated_monthly_savings == 160.00
        assert rec.estimated_annual_savings == 1920.00
        assert rec.projected_cost_monthly == 0.0
        assert (
            rec.is_guaranteed is False
        )  # Net savings cannot be guaranteed without commitment info
        assert rec.confidence_score >= 0.85
        assert any("idle threshold" in ev for ev in rec.evidence)
        assert any("Commitment coverage information is unavailable" in ev for ev in rec.evidence)
        assert any("Direct resource-level billing elimination" in ev for ev in rec.evidence)

    def test_idle_compute_with_active_commitment(self):
        """Idle compute where Reserved Instance or Savings Plan applies: classified as estimated."""
        res = CloudResource(
            resource_id="i-test-idle-ri",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.COMPUTE,
            region="us-east-1",
            monthly_cost=160.00,
            currency="USD",
            specs={"instance_type": "m5.large", "pricing_model": "reserved"},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                avg_cpu_pct=0.3,
                peak_cpu_pct=1.5,
                network_in_bytes=0.0,
                network_out_bytes=0.0,
                is_attached=True,
            )
            for i in range(7)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is not None
        assert rec.action == OptimizationAction.TERMINATE
        assert rec.is_guaranteed is False
        assert any("Instance is associated with a commitment" in ev for ev in rec.evidence)

    def test_idle_compute_confirmed_uncommitted_on_demand(self):
        """Idle compute confirmed on-demand with no commitments: classified as guaranteed."""
        res = CloudResource(
            resource_id="i-test-idle-ondemand",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.COMPUTE,
            region="us-east-1",
            monthly_cost=160.00,
            currency="USD",
            specs={
                "instance_type": "m5.large",
                "pricing_model": "on-demand",
                "has_active_commitment": False,
            },
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                avg_cpu_pct=0.4,
                peak_cpu_pct=2.1,
                network_in_bytes=0.0,
                network_out_bytes=0.0,
                is_attached=True,
            )
            for i in range(10)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is not None
        assert rec.action == OptimizationAction.TERMINATE
        assert rec.is_guaranteed is True
        assert any("Instance is confirmed on-demand/spot" in ev for ev in rec.evidence)

    def test_underutilized_compute_scenario(self):
        res = CloudResource(
            resource_id="vm-test-underutilized-02",
            provider=CloudProvider.AZURE,
            account_id="sub-1",
            service_name="Virtual Machines",
            category=ResourceCategory.COMPUTE,
            region="eastus",
            monthly_cost=300.00,
            currency="USD",
            specs={"instance_type": "Standard_D8s_v5"},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                avg_cpu_pct=8.0,
                peak_cpu_pct=22.0,
                network_in_bytes=500000.0,
                network_out_bytes=600000.0,
                is_attached=True,
            )
            for i in range(10)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is not None
        assert rec.action == OptimizationAction.DOWNSIZE
        assert rec.estimated_monthly_savings == 150.00  # 50%
        assert rec.projected_cost_monthly == 150.00
        assert rec.is_guaranteed is False
        assert rec.confidence_score >= 0.75
        assert "Standard_D8s_v5" in (rec.recommended_specs or "")

    def test_unattached_storage_volume_scenario(self):
        res = CloudResource(
            resource_id="vol-test-unattached-03",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEBS",
            category=ResourceCategory.STORAGE,
            monthly_cost=75.00,
            currency="USD",
            specs={"volume_type": "gp3", "size_gb": 500},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                disk_read_bytes_per_sec=0.0,
                disk_write_bytes_per_sec=0.0,
                is_attached=False,
            )
            for i in range(7)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is not None
        assert rec.action == OptimizationAction.CLEANUP_ORPHAN
        assert rec.risk == OptimizationRisk.LOW
        assert rec.estimated_monthly_savings == 75.00
        assert rec.projected_cost_monthly == 0.0
        assert rec.is_guaranteed is True
        assert rec.confidence_score == 0.95

    def test_overprovisioned_database_scenario(self):
        res = CloudResource(
            resource_id="db-test-rds-04",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonRDS",
            category=ResourceCategory.DATABASE,
            monthly_cost=400.00,
            currency="USD",
            specs={"instance_class": "db.r5.2xlarge"},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                avg_cpu_pct=6.0,
                peak_cpu_pct=15.0,
                avg_memory_pct=10.0,
                is_attached=True,
            )
            for i in range(14)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is not None
        assert rec.action == OptimizationAction.DOWNSIZE
        assert rec.risk == OptimizationRisk.MEDIUM
        assert rec.estimated_monthly_savings == 160.00  # 40%
        assert rec.projected_cost_monthly == 240.00
        assert rec.is_guaranteed is False

    def test_orphaned_static_ip_scenario(self):
        res = CloudResource(
            resource_id="eipalloc-test-ip-05",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.NETWORK,
            monthly_cost=3.60,
            currency="USD",
            specs={"public_ip": "54.210.1.2"},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                network_in_bytes=0.0,
                network_out_bytes=0.0,
                is_attached=False,
            )
            for i in range(7)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is not None
        assert rec.action == OptimizationAction.CLEANUP_ORPHAN
        assert rec.risk == OptimizationRisk.LOW
        assert rec.estimated_monthly_savings == 3.60
        assert rec.confidence_score == 0.98
        assert rec.is_guaranteed is True

    def test_healthy_workload_zero_false_positives(self):
        res = CloudResource(
            resource_id="i-test-healthy-06",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.COMPUTE,
            monthly_cost=250.00,
            currency="USD",
            specs={"instance_type": "m5.2xlarge"},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                avg_cpu_pct=55.0,
                peak_cpu_pct=85.0,
                network_in_bytes=1000000.0,
                network_out_bytes=2000000.0,
                is_attached=True,
            )
            for i in range(14)
        ]

        rec = self.optimizer.analyze_resource(res, telemetry)
        assert rec is None

    def test_missing_telemetry_safe_handling(self):
        res = CloudResource(
            resource_id="i-no-telemetry",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.COMPUTE,
            monthly_cost=100.00,
        )
        assert self.optimizer.analyze_resource(res, None) is None
        assert self.optimizer.analyze_resource(res, []) is None

    def test_unattached_storage_insufficient_telemetry_returns_none(self):
        """F-03: Orphan detection requires telemetry and does not infer from inventory specs alone."""
        res = CloudResource(
            resource_id="vol-test-unverified",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEBS",
            category=ResourceCategory.STORAGE,
            monthly_cost=80.00,
            specs={"status": "available", "is_attached": False},
        )
        assert self.optimizer.analyze_resource(res, None) is None
        assert self.optimizer.analyze_resource(res, []) is None

    def test_attached_storage_volume_not_flagged(self):
        """Storage volume actively attached in telemetry is not flagged for orphan cleanup."""
        res = CloudResource(
            resource_id="vol-attached-active",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEBS",
            category=ResourceCategory.STORAGE,
            monthly_cost=80.00,
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                is_attached=True,
            )
            for i in range(5)
        ]
        assert self.optimizer.analyze_resource(res, telemetry) is None

    def test_orphaned_ip_insufficient_telemetry_returns_none(self):
        """F-03: IP orphan detection requires valid telemetry observation."""
        res = CloudResource(
            resource_id="eipalloc-unverified",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.NETWORK,
            monthly_cost=3.60,
            specs={"allocation_id": "eipalloc-unverified", "is_attached": False},
        )
        assert self.optimizer.analyze_resource(res, None) is None
        assert self.optimizer.analyze_resource(res, []) is None

    def test_attached_ip_not_flagged(self):
        """Public IP actively attached in telemetry is not flagged as orphaned."""
        res = CloudResource(
            resource_id="eipalloc-active",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.NETWORK,
            monthly_cost=3.60,
            specs={"allocation_id": "eipalloc-active"},
        )
        telemetry = [
            ResourceUtilization(
                resource_id=res.resource_id,
                service_name=res.service_name,
                timestamp=self.now - datetime.timedelta(days=i),
                is_attached=True,
                network_in_bytes=10000.0,
            )
            for i in range(5)
        ]
        assert self.optimizer.analyze_resource(res, telemetry) is None


class TestInventoryOptimization:
    def test_synthetic_inventory_optimization(self):
        resources, utilization_map = get_synthetic_resource_inventory()
        optimizer = ResourceOptimizer()
        summary = optimizer.optimize_inventory(resources, utilization_map)

        assert summary.recommendations_count == 5
        assert summary.total_monthly_savings == pytest.approx(598.10)
        assert summary.total_annual_savings == pytest.approx(598.10 * 12.0)
        # Guaranteed: $80 (EBS) + $3.60 (EIP) = $83.60
        assert summary.guaranteed_monthly_savings == pytest.approx(83.60)
        assert summary.guaranteed_annual_savings == pytest.approx(83.60 * 12.0)
        # Estimated: $145 (EC2) + $137.50 (Azure VM) + $232 (RDS) = $514.50
        assert summary.estimated_monthly_savings == pytest.approx(514.50)
        assert summary.estimated_annual_savings == pytest.approx(514.50 * 12.0)
        assert (
            summary.guaranteed_annual_savings + summary.estimated_annual_savings
            == pytest.approx(summary.total_annual_savings)
        )
        assert "TERMINATE" in summary.recommendations_by_action
        assert "DOWNSIZE" in summary.recommendations_by_action
        assert "CLEANUP_ORPHAN" in summary.recommendations_by_action

    def test_empty_inventory_returns_empty_summary(self):
        optimizer = ResourceOptimizer()
        summary = optimizer.optimize_inventory([], {})
        assert summary.recommendations_count == 0
        assert summary.total_monthly_savings == 0.0
        assert summary.total_annual_savings == 0.0
        assert summary.guaranteed_monthly_savings == 0.0
        assert summary.guaranteed_annual_savings == 0.0
        assert summary.estimated_monthly_savings == 0.0
        assert summary.estimated_annual_savings == 0.0
        assert summary.recommendations == []

    def test_currency_mismatch_raises_error(self):
        r1 = CloudResource(
            resource_id="r1",
            provider=CloudProvider.AWS,
            account_id="1",
            service_name="EC2",
            category=ResourceCategory.COMPUTE,
            monthly_cost=100.0,
            currency="USD",
        )
        r2 = CloudResource(
            resource_id="r2",
            provider=CloudProvider.AWS,
            account_id="1",
            service_name="EC2",
            category=ResourceCategory.COMPUTE,
            monthly_cost=100.0,
            currency="EUR",
        )
        optimizer = ResourceOptimizer()
        with pytest.raises(CurrencyMismatchError):
            optimizer.optimize_inventory([r1, r2], {})

    def test_filters_by_category_and_risk(self):
        resources, utilization_map = get_synthetic_resource_inventory()
        optimizer = ResourceOptimizer()

        # Filter by storage category
        storage_summary = optimizer.optimize_inventory(
            resources, utilization_map, category_filter=ResourceCategory.STORAGE
        )
        assert storage_summary.recommendations_count == 1
        assert storage_summary.recommendations[0].action == OptimizationAction.CLEANUP_ORPHAN

        # Filter by minimum savings ($100/mo)
        high_value_summary = optimizer.optimize_inventory(
            resources, utilization_map, min_savings=100.0
        )
        for r in high_value_summary.recommendations:
            assert r.estimated_monthly_savings >= 100.0
