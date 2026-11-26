# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for FinOps data models, currency handling, and validation bounds."""

import datetime
import pytest
from pydantic import ValidationError

from devops_ai.finops.models import (
    AnomalySeverity,
    AnomalySummary,
    BillingPeriod,
    BillingRecord,
    CloudProvider,
    CloudResource,
    CostAnomaly,
    CostForecast,
    CostSummary,
    FinOpsAdvisoryRecommendation,
    FinOpsAdvisoryReport,
    ForecastMethod,
    ForecastPoint,
    ForecastSummary,
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

    def test_cost_anomaly_extended_fields(self):
        anom = CostAnomaly(
            detected_date=datetime.date(2026, 11, 20),
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            region="us-east-1",
            resource_id="i-test-anomaly",
            actual_cost=450.0,
            expected_cost=150.0,
            cost_delta=300.0,
            percentage_increase=200.0,
            z_score=4.2,
            severity=AnomalySeverity.CRITICAL,
            confidence_score=0.95,
            detection_method="z_score",
            historical_observation_count=30,
            driver_resources=["i-test-anomaly"],
            explanation="Unusual batch surge",
            evidence=["Z-score 4.2 > 2.5"],
        )
        assert anom.provider == CloudProvider.AWS
        assert anom.account_id == "123456789012"
        assert anom.region == "us-east-1"
        assert anom.resource_id == "i-test-anomaly"
        assert anom.confidence_score == 0.95
        assert anom.historical_observation_count == 30

    def test_anomaly_summary_model(self):
        summary = AnomalySummary(
            total_anomalies=2,
            total_excess_spend=500.0,
            currency="USD",
            anomalies_by_severity={"CRITICAL": 1, "HIGH": 1},
            anomalies_by_service={"AmazonEC2": 2},
            period_start=datetime.date(2026, 10, 1),
            period_end=datetime.date(2026, 10, 31),
        )
        assert summary.total_anomalies == 2
        assert summary.total_excess_spend == 500.0
        assert summary.anomalies_by_severity["CRITICAL"] == 1

    def test_forecast_point_quantiles_sync_and_validation(self):
        # 1. Automatic synchronization when p10, p50, p90 omitted
        pt1 = ForecastPoint(
            date=datetime.date(2026, 12, 1),
            projected_cost=100.0,
            lower_bound=80.0,
            upper_bound=120.0,
        )
        assert pt1.p10 == 80.0
        assert pt1.p50 == 100.0
        assert pt1.p90 == 120.0

        # 2. Invariant enforcement: 0 <= p10 <= p50 <= p90
        pt2 = ForecastPoint(
            date=datetime.date(2026, 12, 2),
            projected_cost=100.0,
            lower_bound=110.0,  # lower_bound > projected_cost
            upper_bound=90.0,  # upper_bound < projected_cost
        )
        # Validator clamps p10 and p90 to satisfy p10 <= p50 <= p90
        assert pt2.p10 <= pt2.p50 <= pt2.p90
        assert pt2.p10 == 100.0
        assert pt2.p50 == 100.0
        assert pt2.p90 == 100.0

    def test_forecast_summary_model(self):
        point = ForecastPoint(
            date=datetime.date(2026, 12, 1),
            projected_cost=100.0,
            lower_bound=80.0,
            upper_bound=120.0,
        )
        fcst = CostForecast(
            days_projected=1,
            method=ForecastMethod.LINEAR_REGRESSION.value,
            total_projected_spend=100.0,
            total_p10_spend=80.0,
            total_p90_spend=120.0,
            currency="USD",
            trend="increasing",
            points=[point],
        )
        summary = ForecastSummary(
            forecast_id=fcst.forecast_id,
            currency="USD",
            horizon_days=1,
            forecast_start_date=datetime.date(2026, 12, 1),
            forecast_end_date=datetime.date(2026, 12, 1),
            total_projected_cost=100.0,
            total_p10_cost=80.0,
            total_p90_cost=120.0,
            daily_average_cost=100.0,
            trend="increasing",
            method=fcst.method,
            confidence_level=0.90,
            historical_days_analyzed=30,
            forecast=fcst,
        )
        assert summary.horizon_days == 1
        assert summary.daily_average_cost == 100.0
        assert summary.total_p10_cost == 80.0
        assert summary.total_p90_cost == 120.0

    def test_finops_advisory_recommendation_model(self):
        rec = FinOpsAdvisoryRecommendation(
            title="Clean up unattached EBS volumes",
            category="storage",
            priority="high",
            savings_type="guaranteed",
            potential_monthly_savings=150.0,
            confidence_score=0.95,
            action="Delete unattached gp3 volumes vol-001 and vol-002",
            evidence=[
                "Volume vol-001 unattached for 45 days",
                "Volume vol-002 unattached for 30 days",
            ],
            affected_resources=["vol-001", "vol-002"],
        )
        assert rec.title == "Clean up unattached EBS volumes"
        assert rec.potential_monthly_savings == 150.0
        assert rec.potential_annual_savings == 1800.0
        assert rec.savings_type == "guaranteed"
        assert len(rec.affected_resources) == 2

        # Invariant and bounds validation
        with pytest.raises(ValidationError):
            FinOpsAdvisoryRecommendation(
                title="Negative savings test",
                category="compute",
                action="Test",
                potential_monthly_savings=-10.0,
            )

        with pytest.raises(ValidationError):
            FinOpsAdvisoryRecommendation(
                title="Confidence out of bounds",
                category="compute",
                action="Test",
                confidence_score=1.5,
            )

        with pytest.raises(ValidationError):
            # Extra fields forbidden
            FinOpsAdvisoryRecommendation(
                title="Extra fields test",
                category="compute",
                action="Test",
                unknown_extra_field="invalid",
            )

    def test_finops_advisory_report_model(self):
        rec1 = FinOpsAdvisoryRecommendation(
            title="Clean unattached disks",
            category="storage",
            savings_type="guaranteed",
            potential_monthly_savings=100.0,
            action="Delete disk-1",
        )
        rec2 = FinOpsAdvisoryRecommendation(
            title="Downsize oversized instances",
            category="compute",
            savings_type="estimated",
            potential_monthly_savings=250.0,
            action="Downsize i-001 to t3.medium",
        )

        report = FinOpsAdvisoryReport(
            executive_summary="Identified substantial cost optimization opportunities.",
            recommendations=[rec1, rec2],
            anomalies_detected=1,
            forecast_trend="increasing",
            provider="ollama",
            model="llama3.2:1b",
        )

        # Automatic calculation of totals
        assert report.guaranteed_monthly_savings == 100.0
        assert report.estimated_monthly_savings == 250.0
        assert report.total_monthly_savings == 350.0
        assert report.guaranteed_annual_savings == 1200.0
        assert report.estimated_annual_savings == 3000.0
        assert report.total_annual_savings == 4200.0
        assert "Automated cloud mutations are disabled" in report.disclaimer

        # Extra fields forbidden
        with pytest.raises(ValidationError):
            FinOpsAdvisoryReport(
                executive_summary="Summary",
                unexpected_field="forbidden",
            )

    def test_finops_advisory_report_historical_excess_spend_and_currency(self):
        """Verify historical_excess_spend is tracked as separate risk exposure without affecting savings (F-01, F-03)."""
        anom_rec = FinOpsAdvisoryRecommendation(
            title="Investigate anomaly spikes",
            category="anomaly",
            savings_type="estimated",
            potential_monthly_savings=0.0,
            currency="EUR",
            action="Audit root causes",
            evidence=["Historical spike: EUR 450.00"],
        )
        report = FinOpsAdvisoryReport(
            executive_summary="European spending advisory report",
            recommendations=[anom_rec],
            currency="EUR",
            anomalies_detected=2,
            historical_excess_spend=450.0,
        )
        assert report.currency == "EUR"
        assert report.historical_excess_spend == 450.0
        assert report.total_monthly_savings == 0.0
        assert report.total_annual_savings == 0.0
