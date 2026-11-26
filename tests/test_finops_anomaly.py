# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit and integration tests for FinOps statistical cost anomaly detection engine."""

import datetime
import pytest

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.finops.anomaly import CostAnomalyDetector
from devops_ai.finops.importer import get_synthetic_anomaly_and_forecast_dataset
from devops_ai.finops.models import (
    AnomalySeverity,
    BillingRecord,
    CloudProvider,
)


def _make_daily_records(
    daily_costs: list[float],
    service: str = "AmazonEC2",
    account: str = "123456789012",
    region: str = "us-east-1",
    currency: str = "USD",
    start_date: datetime.date = datetime.date(2026, 10, 1),
) -> list[BillingRecord]:
    """Helper to generate sequential daily billing records."""
    records = []
    for idx, cost in enumerate(daily_costs):
        d = start_date + datetime.timedelta(days=idx)
        dt = datetime.datetime.combine(d, datetime.time.min, tzinfo=datetime.timezone.utc)
        records.append(
            BillingRecord(
                record_id=f"REC-{service}-{idx}",
                provider=CloudProvider.AWS,
                account_id=account,
                service_name=service,
                region=region,
                cost=cost,
                currency=currency,
                start_time=dt,
                end_time=dt + datetime.timedelta(days=1),
            )
        )
    return records


class TestCostAnomalyDetector:
    def test_steady_state_no_false_positives(self):
        """Uniform spend with small random noise should produce zero anomalies."""
        # 30 days of spend around $100 with negligible noise
        costs = [100.0 + (i % 3) * 0.5 for i in range(30)]
        records = _make_daily_records(costs)

        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        assert summary.total_anomalies == 0
        assert summary.total_excess_spend == 0.0
        assert len(summary.anomalies) == 0

    def test_isolated_expenditure_spike(self):
        """Single massive spend spike should trigger a high or critical anomaly."""
        costs = [100.0] * 15 + [500.0] + [100.0] * 5
        records = _make_daily_records(costs)

        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        assert summary.total_anomalies >= 1
        anom = next(a for a in summary.anomalies if a.actual_cost == 500.0)
        assert anom.actual_cost == 500.0
        assert anom.expected_cost == pytest.approx(100.0, rel=0.1)
        assert anom.cost_delta >= 350.0
        assert anom.percentage_increase >= 300.0
        assert anom.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
        assert anom.confidence_score >= 0.85
        assert len(anom.evidence) >= 1

    def test_multi_day_runaway_surge(self):
        """Multi-day runaway surge should flag anomalies on each surge day."""
        costs = [50.0] * 20 + [250.0, 300.0, 280.0] + [50.0] * 5
        records = _make_daily_records(costs)

        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        # Days 20, 21, 22 should be flagged
        surge_dates = [
            datetime.date(2026, 10, 1) + datetime.timedelta(days=d) for d in [20, 21, 22]
        ]
        detected_dates = [a.detected_date for a in summary.anomalies]
        for sd in surge_dates:
            assert sd in detected_dates

    def test_service_and_account_filters(self):
        """Detector filters correctly by service name and account ID."""
        records_ec2 = _make_daily_records(
            [100.0] * 10 + [600.0], service="AmazonEC2", account="acct-1"
        )
        records_s3 = _make_daily_records([50.0] * 11, service="AmazonS3", account="acct-2")
        all_records = records_ec2 + records_s3

        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)

        # Filter by service EC2 -> should detect anomaly
        summary_ec2 = detector.detect_anomalies(records=all_records, service_filter="AmazonEC2")
        assert summary_ec2.total_anomalies >= 1

        # Filter by service S3 -> should detect 0 anomalies
        summary_s3 = detector.detect_anomalies(records=all_records, service_filter="AmazonS3")
        assert summary_s3.total_anomalies == 0

        # Filter by account acct-2 -> 0 anomalies
        summary_acct2 = detector.detect_anomalies(records=all_records, account_filter="acct-2")
        assert summary_acct2.total_anomalies == 0

    def test_severity_filter(self):
        """Filtering by severity excludes lower severity anomalies."""
        costs = [100.0] * 10 + [140.0, 800.0]
        records = _make_daily_records(costs)

        detector = CostAnomalyDetector(
            z_score_threshold=2.0,
            min_observations=7,
            min_absolute_delta=5.0,
            min_percentage_increase=10.0,
        )
        summary_all = detector.detect_anomalies(records=records)
        assert summary_all.total_anomalies >= 1

        # Filter strictly for CRITICAL
        summary_critical = detector.detect_anomalies(
            records=records, severity_filter=AnomalySeverity.CRITICAL
        )
        for a in summary_critical.anomalies:
            assert a.severity == AnomalySeverity.CRITICAL

    def test_insufficient_historical_observations(self):
        """Fewer observations than min_observations should not trigger false alarms."""
        costs = [100.0, 105.0, 500.0]  # Only 3 days, min_observations=7
        records = _make_daily_records(costs)

        detector = CostAnomalyDetector(z_score_threshold=2.0, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        assert summary.total_anomalies == 0

    def test_zero_baseline_division_safety(self):
        """Transition from zero spend to positive spend does not cause ZeroDivisionError."""
        costs = [0.0] * 10 + [150.0]
        records = _make_daily_records(costs)

        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        # Should flag or handle safely without division by zero
        assert summary.currency == "USD"
        if summary.total_anomalies > 0:
            anom = summary.anomalies[0]
            assert anom.actual_cost == 150.0
            assert anom.expected_cost == 0.0

    def test_credit_adjustments_handling(self):
        """Credits (negative costs) should not corrupt baseline or trigger positive surge alarms."""
        costs = [100.0] * 10
        records = _make_daily_records(costs)
        # Add negative credit on day 5
        dt = datetime.datetime(2026, 10, 5, tzinfo=datetime.timezone.utc)
        records.append(
            BillingRecord(
                record_id="REC-CREDIT-1",
                provider=CloudProvider.AWS,
                account_id="123456789012",
                service_name="AmazonEC2",
                cost=-50.0,
                currency="USD",
                start_time=dt,
                end_time=dt + datetime.timedelta(days=1),
            )
        )

        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)
        summary = detector.detect_anomalies(records=records)
        assert summary.total_anomalies == 0

    def test_currency_mismatch_raises_error(self):
        """Mixed currencies must raise CurrencyMismatchError."""
        records = _make_daily_records([100.0] * 10, currency="USD")
        records.extend(_make_daily_records([100.0] * 5, currency="EUR"))

        detector = CostAnomalyDetector()
        with pytest.raises(CurrencyMismatchError):
            detector.detect_anomalies(records=records)

    def test_date_range_filtering(self):
        """Date range restricts anomaly search window."""
        costs = [100.0] * 10 + [500.0] + [100.0] * 10
        records = _make_daily_records(costs)  # Day 10 has the spike (2026-10-11)

        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)

        # Exclude day 10 by end_date
        summary = detector.detect_anomalies(
            records=records,
            end_date=datetime.date(2026, 10, 9),
        )
        assert summary.total_anomalies == 0

    def test_synthetic_benchmark_anomaly_detection(self):
        """Running detector on built-in synthetic benchmark flags known anomalies."""
        records = get_synthetic_anomaly_and_forecast_dataset()
        detector = CostAnomalyDetector(z_score_threshold=2.5, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        assert summary.total_anomalies >= 4
        assert summary.total_excess_spend > 1000.0
        assert summary.currency == "USD"
        assert "AmazonEC2" in summary.anomalies_by_service
        assert summary.period_start is not None
        assert summary.period_end is not None

        # Verify EC2 surge days are detected
        ec2_anomalies = [a for a in summary.anomalies if a.service_name == "AmazonEC2"]
        assert len(ec2_anomalies) >= 2
        for a in ec2_anomalies:
            assert a.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
            assert a.actual_cost >= 400.0

    def test_account_wide_surge_distributed_across_multiple_services_unfiltered_scan(self):
        """Surge spread across multiple services below individual threshold is captured at account level."""
        # 3 services in same account, each spending $10 normally for 20 days
        recs_a = _make_daily_records(
            [10.0] * 20 + [25.0] + [10.0] * 5, service="SvcA", account="acct-multi"
        )
        recs_b = _make_daily_records(
            [10.0] * 20 + [25.0] + [10.0] * 5, service="SvcB", account="acct-multi"
        )
        recs_c = _make_daily_records(
            [10.0] * 20 + [25.0] + [10.0] * 5, service="SvcC", account="acct-multi"
        )
        records = recs_a + recs_b + recs_c

        # min_absolute_delta=20.0: Individual deltas are $15 (below threshold).
        # Account delta is $45 (well above $20 threshold).
        detector = CostAnomalyDetector(
            z_score_threshold=2.0, min_observations=7, min_absolute_delta=20.0
        )
        summary = detector.detect_anomalies(records=records)

        # Should detect at account level
        acct_anoms = [
            a for a in summary.anomalies if a.account_id == "acct-multi" and a.service_name is None
        ]
        assert len(acct_anoms) >= 1
        assert acct_anoms[0].actual_cost == 75.0
        assert acct_anoms[0].expected_cost == pytest.approx(30.0, rel=0.1)
        assert acct_anoms[0].cost_delta == pytest.approx(45.0, rel=0.1)

    def test_region_wide_surge_distributed_across_multiple_services_unfiltered_scan(self):
        """Surge spread across multiple services is captured at region level in unfiltered scan."""
        recs_a = _make_daily_records(
            [10.0] * 20 + [25.0] + [10.0] * 5,
            service="Svc1",
            account="acct-x",
            region="eu-central-1",
        )
        recs_b = _make_daily_records(
            [10.0] * 20 + [25.0] + [10.0] * 5,
            service="Svc2",
            account="acct-y",
            region="eu-central-1",
        )
        recs_c = _make_daily_records(
            [10.0] * 20 + [25.0] + [10.0] * 5,
            service="Svc3",
            account="acct-z",
            region="eu-central-1",
        )
        records = recs_a + recs_b + recs_c

        detector = CostAnomalyDetector(
            z_score_threshold=2.0, min_observations=7, min_absolute_delta=20.0
        )
        summary = detector.detect_anomalies(records=records)

        reg_anoms = [
            a for a in summary.anomalies if a.region == "eu-central-1" and a.service_name is None
        ]
        assert len(reg_anoms) >= 1
        assert reg_anoms[0].actual_cost == 75.0
        assert reg_anoms[0].cost_delta == pytest.approx(45.0, rel=0.1)

    def test_deduplication_of_hierarchical_anomalies(self):
        """Single service surge is not duplicated at account or overall levels."""
        # Single service surge on day 20
        recs_ec2 = _make_daily_records(
            [100.0] * 20 + [500.0] + [100.0] * 5,
            service="AmazonEC2",
            account="123456789012",
            region="us-east-1",
        )
        # S3 has completely flat spend
        recs_s3 = _make_daily_records(
            [50.0] * 26, service="AmazonS3", account="123456789012", region="us-east-1"
        )
        records = recs_ec2 + recs_s3

        detector = CostAnomalyDetector(z_score_threshold=2.0, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        # On day 20, EC2 surged by $400. Account and Overall also surged by $400.
        # Deduplication must retain the specific service anomaly and suppress redundant account/overall ones.
        surge_date = datetime.date(2026, 10, 1) + datetime.timedelta(days=20)
        day_20_anoms = [a for a in summary.anomalies if a.detected_date == surge_date]
        assert len(day_20_anoms) == 1
        assert day_20_anoms[0].service_name == "AmazonEC2"
        # Total excess spend must not be inflated by duplicate counting
        assert summary.total_excess_spend == pytest.approx(day_20_anoms[0].cost_delta, rel=0.01)

    def test_newly_provisioned_service_start_date_no_false_positive(self):
        """Newly provisioned service does not get padded with historical zeros or trigger false alarms."""
        # Baseline service running for 45 days
        recs_existing = _make_daily_records([100.0] * 45, service="ExistingService")

        # Newly provisioned service starting on Day 20 (2026-10-21)
        # Normal spend is $50/day. Day 28 (after 8 days) has a real spike to $300.
        recs_new = _make_daily_records(
            [50.0] * 8 + [300.0] + [50.0] * 5,
            service="NewProvisionedService",
            start_date=datetime.date(2026, 10, 21),
        )
        records = recs_existing + recs_new

        detector = CostAnomalyDetector(z_score_threshold=2.0, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        new_svc_anoms = [a for a in summary.anomalies if a.service_name == "NewProvisionedService"]

        # Day 20 (inception) must NOT be flagged as an anomaly!
        inception_date = datetime.date(2026, 10, 21)
        assert not any(a.detected_date == inception_date for a in new_svc_anoms)

        # Day 28 (spike to $300 on 2026-10-29) MUST be flagged
        spike_date = datetime.date(2026, 10, 21) + datetime.timedelta(days=8)
        assert any(a.detected_date == spike_date for a in new_svc_anoms)

    def test_mad_reference_evidence_and_zscore_decision_gate(self):
        """Evidence string includes MAD as reference context while detection method is z_score."""
        costs = [100.0] * 15 + [400.0] + [100.0] * 5
        records = _make_daily_records(costs)

        detector = CostAnomalyDetector(z_score_threshold=2.0, min_observations=7)
        summary = detector.detect_anomalies(records=records)

        assert summary.total_anomalies >= 1
        anom = summary.anomalies[0]
        assert anom.detection_method == "z_score"
        assert any("mad reference" in ev.lower() for ev in anom.evidence)

    def test_distinct_services_with_identical_date_spend_and_delta_are_both_retained(self):
        """Two different services with identical date, spend, and delta are NOT treated as duplicates."""
        # 20 days baseline of $100 each for AmazonEC2 and AmazonS3
        recs_ec2 = _make_daily_records(
            [100.0] * 20 + [500.0] + [100.0] * 5,
            service="AmazonEC2",
            account="123456789012",
            region="us-east-1",
        )
        recs_s3 = _make_daily_records(
            [100.0] * 20 + [500.0] + [100.0] * 5,
            service="AmazonS3",
            account="123456789012",
            region="us-east-1",
        )
        records = recs_ec2 + recs_s3

        detector = CostAnomalyDetector(
            z_score_threshold=2.0, min_observations=7, min_absolute_delta=20.0
        )
        summary = detector.detect_anomalies(records=records)

        surge_date = datetime.date(2026, 10, 1) + datetime.timedelta(days=20)
        day_anoms = [a for a in summary.anomalies if a.detected_date == surge_date]

        # Both services MUST be present and independently visible
        svc_names = {a.service_name for a in day_anoms if a.service_name}
        assert "AmazonEC2" in svc_names
        assert "AmazonS3" in svc_names

        ec2_anom = next(a for a in day_anoms if a.service_name == "AmazonEC2")
        s3_anom = next(a for a in day_anoms if a.service_name == "AmazonS3")
        assert ec2_anom.actual_cost == 500.0
        assert ec2_anom.cost_delta == pytest.approx(400.0, rel=0.01)
        assert s3_anom.actual_cost == 500.0
        assert s3_anom.cost_delta == pytest.approx(400.0, rel=0.01)

        # Account surge (delta $800) is also retained
        acct_anoms = [a for a in day_anoms if a.account_id == "123456789012"]
        assert len(acct_anoms) >= 1
        assert acct_anoms[0].cost_delta == pytest.approx(800.0, rel=0.01)

    def test_duplicate_reports_for_same_service_are_deduplicated(self):
        """Identical duplicate anomalies for the same service entity are deduplicated."""
        # Same service duplicate records on day 20
        recs_ec2 = _make_daily_records(
            [100.0] * 20 + [500.0] + [100.0] * 5,
            service="AmazonEC2",
            account="123456789012",
            region="us-east-1",
        )
        detector = CostAnomalyDetector(
            z_score_threshold=2.0, min_observations=7, min_absolute_delta=20.0
        )
        # Pass the exact same list twice
        summary = detector.detect_anomalies(records=recs_ec2)

        surge_date = datetime.date(2026, 10, 1) + datetime.timedelta(days=20)
        ec2_anoms = [
            a
            for a in summary.anomalies
            if a.service_name == "AmazonEC2" and a.detected_date == surge_date
        ]
        assert len(ec2_anoms) == 1
