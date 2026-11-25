# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for FinOps data models, currency handling, and validation bounds."""

import datetime
import pytest
from pydantic import ValidationError

from devops_ai.finops.models import (
    AnomalySeverity,
    BillingPeriod,
    BillingRecord,
    CloudProvider,
    CloudResource,
    CostAnomaly,
    CostForecast,
    CostSummary,
    ForecastPoint,
    OptimizationAction,
    OptimizationRecommendation,
    OptimizationRisk,
    OptimizationSummary,
    ResourceCategory,
    ResourceUtilization,
)


class TestBillingRecordModel:
    def test_valid_billing_record(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        record = BillingRecord(
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            cost=45.50,
            currency="USD",
            start_time=now,
            end_time=now + datetime.timedelta(hours=1),
        )
        assert record.record_id.startswith("REC-")
        assert record.cost == 45.50
        assert record.currency == "USD"
        assert record.is_estimated is False

    def test_signed_cost_and_credits(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        # Positive charge
        charge = BillingRecord(
            provider=CloudProvider.AWS,
            account_id="123",
            service_name="EC2",
            cost=50.0,
            start_time=now,
            end_time=now,
        )
        assert charge.cost == 50.0
        assert charge.is_credit is False

        # Negative credit or refund
        credit = BillingRecord(
            provider=CloudProvider.AWS,
            account_id="123",
            service_name="EC2",
            cost=-25.50,
            start_time=now,
            end_time=now,
        )
        assert credit.cost == -25.50
        assert credit.is_credit is True

    def test_invalid_currency_code_length(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        with pytest.raises(ValidationError):
            BillingRecord(
                provider=CloudProvider.GCP,
                account_id="proj-1",
                service_name="GCE",
                cost=10.0,
                currency="US",  # < 3 chars
                start_time=now,
                end_time=now,
            )


class TestCloudResourceModel:
    def test_valid_cloud_resource(self):
        res = CloudResource(
            resource_id="i-0abcd1234ef567890",
            resource_name="prod-api-server-1",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            category=ResourceCategory.COMPUTE,
            region="us-east-1",
            monthly_cost=150.0,
            currency="USD",
            specs={"instance_type": "m5.large", "vcpu": 2},
            tags={"Environment": "production"},
        )
        assert res.resource_id == "i-0abcd1234ef567890"
        assert res.category == ResourceCategory.COMPUTE
        assert res.specs["instance_type"] == "m5.large"

    def test_negative_monthly_cost_rejected(self):
        with pytest.raises(ValidationError):
            CloudResource(
                resource_id="i-12345",
                provider=CloudProvider.AWS,
                account_id="12345",
                service_name="EC2",
                category=ResourceCategory.COMPUTE,
                monthly_cost=-10.0,
            )


class TestCostSummaryModel:
    def test_valid_cost_summary(self):
        summary = CostSummary(
            total_cost=2450.00,
            gross_cost=2500.00,
            credit_cost=-50.00,
            currency="USD",
            total_records=100,
            service_costs={"AmazonEC2": 1500.0, "AmazonS3": 950.0},
            daily_costs={"2026-11-20": 1250.00, "2026-11-21": 1200.00},
        )
        assert summary.total_cost == 2450.00
        assert summary.gross_cost == 2500.00
        assert summary.credit_cost == -50.00
        assert summary.currency == "USD"
        assert summary.total_records == 100
        assert summary.service_costs["AmazonEC2"] == 1500.0

    def test_credit_cost_must_be_non_positive(self):
        with pytest.raises(ValidationError):
            CostSummary(
                total_cost=100.0,
                credit_cost=50.0,  # positive credit_cost rejected by le=0.0
            )


class TestBillingPeriodModel:
    def test_duration_days_calculation(self):
        t1 = datetime.datetime(2026, 11, 1, 0, 0, tzinfo=datetime.timezone.utc)
        t2 = datetime.datetime(2026, 11, 11, 0, 0, tzinfo=datetime.timezone.utc)
        period = BillingPeriod(start_time=t1, end_time=t2)
        assert period.duration_days == pytest.approx(10.0)


class TestResourceUtilizationModel:
    def test_valid_utilization(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        util = ResourceUtilization(
            resource_id="i-12345",
            service_name="EC2",
            timestamp=now,
            avg_cpu_pct=15.5,
            peak_cpu_pct=85.0,
        )
        assert util.avg_cpu_pct == 15.5

    def test_cpu_pct_bounds(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        with pytest.raises(ValidationError):
            ResourceUtilization(
                resource_id="i-12345",
                service_name="EC2",
                timestamp=now,
                avg_cpu_pct=105.0,  # > 100%
            )


class TestOptimizationModels:
    def test_recommendation_model(self):
        rec = OptimizationRecommendation(
            resource_id="vol-01234",
            service_name="AmazonEBS",
            action=OptimizationAction.TERMINATE,
            risk=OptimizationRisk.LOW,
            current_cost_monthly=50.0,
            projected_cost_monthly=0.0,
            estimated_monthly_savings=50.0,
            confidence_score=0.95,
            evidence=["Volume unattached for 45 consecutive days"],
        )
        assert rec.action == OptimizationAction.TERMINATE
        assert rec.estimated_monthly_savings == 50.0
        assert rec.confidence_score == 0.95
        assert rec.estimated_annual_savings == 600.0

    def test_optimization_summary_annualized_properties(self):
        summary = OptimizationSummary(
            total_monthly_savings=250.0,
            total_annual_savings=3000.0,
            guaranteed_monthly_savings=100.0,
            estimated_monthly_savings=150.0,
            currency="USD",
            recommendations_count=2,
            recommendations_by_action={"CLEANUP_ORPHAN": 1, "DOWNSIZE": 1},
            recommendations_by_risk={"LOW": 2},
            recommendations=[],
        )
        assert summary.guaranteed_monthly_savings == 100.0
        assert summary.estimated_monthly_savings == 150.0
        assert summary.guaranteed_annual_savings == 1200.0
        assert summary.estimated_annual_savings == 1800.0
        assert (
            summary.guaranteed_annual_savings + summary.estimated_annual_savings
            == summary.total_annual_savings
        )

    def test_optimization_summary_zero_savings(self):
        summary = OptimizationSummary(
            total_monthly_savings=0.0,
            total_annual_savings=0.0,
            guaranteed_monthly_savings=0.0,
            estimated_monthly_savings=0.0,
            currency="EUR",
            recommendations_count=0,
        )
        assert summary.guaranteed_annual_savings == 0.0
        assert summary.estimated_annual_savings == 0.0
        assert summary.total_annual_savings == 0.0
        assert summary.currency == "EUR"


class TestAnomalyAndForecastModels:
    def test_cost_anomaly_model(self):
        anom = CostAnomaly(
            detected_date=datetime.date(2026, 11, 20),
            service_name="AmazonS3",
            actual_cost=450.0,
            expected_cost=150.0,
            cost_delta=300.0,
            percentage_increase=200.0,
            z_score=3.5,
            severity=AnomalySeverity.CRITICAL,
            driver_resources=["bucket-data-transfer"],
            explanation="Spike in outbound data transfer fees",
        )
        assert anom.severity == AnomalySeverity.CRITICAL
        assert anom.z_score == 3.5

    def test_cost_forecast_model(self):
        point = ForecastPoint(
            date=datetime.date(2026, 12, 1),
            projected_cost=100.0,
            lower_bound=90.0,
            upper_bound=110.0,
        )
        fcst = CostForecast(
            days_projected=30,
            method="linear_regression",
            total_projected_spend=3000.0,
            currency="USD",
            points=[point],
        )
        assert fcst.days_projected == 30
        assert len(fcst.points) == 1
