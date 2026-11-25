# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for CostAnalyzer, multi-dimensional expenditure aggregation, and currency consistency."""

import datetime
import pytest

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.finops.analyzer import CostAnalyzer
from devops_ai.finops.importer import get_synthetic_billing_dataset
from devops_ai.finops.models import BillingRecord, CloudProvider


class TestCostAnalyzer:
    def test_empty_records_summary(self):
        analyzer = CostAnalyzer()
        summary = analyzer.summarize([])
        assert summary.total_cost == 0.0
        assert summary.total_records == 0
        assert summary.currency == "USD"
        assert summary.service_costs == {}
        assert summary.daily_costs == {}

    def test_synthetic_dataset_summary(self):
        analyzer = CostAnalyzer()
        records = get_synthetic_billing_dataset()
        summary = analyzer.summarize(records)

        assert summary.total_records == len(records)
        assert summary.total_cost == pytest.approx(1536.25, rel=1e-2)
        assert summary.currency == "USD"
        assert "AmazonEC2" in summary.service_costs
        assert "AmazonRDS" in summary.service_costs
        assert "Virtual Machines" in summary.service_costs
        assert len(summary.top_resources) > 0
        assert summary.period_start is not None
        assert summary.period_end is not None

    def test_currency_mismatch_raises_error(self):
        analyzer = CostAnalyzer()
        now = datetime.datetime.now(datetime.timezone.utc)
        r1 = BillingRecord(
            provider=CloudProvider.AWS,
            account_id="1",
            service_name="EC2",
            cost=10.0,
            currency="USD",
            start_time=now,
            end_time=now,
        )
        r2 = BillingRecord(
            provider=CloudProvider.AWS,
            account_id="1",
            service_name="S3",
            cost=15.0,
            currency="EUR",  # Incompatible currency
            start_time=now,
            end_time=now,
        )

        with pytest.raises(CurrencyMismatchError) as exc_info:
            analyzer.summarize([r1, r2])

        assert exc_info.value.base_currency == "USD"
        assert exc_info.value.found_currency == "EUR"

    def test_filtering_by_service_and_account(self):
        analyzer = CostAnalyzer()
        records = get_synthetic_billing_dataset()

        # Filter by AWS service "AmazonEC2"
        filtered_ec2 = analyzer.filter_records(records, service="AmazonEC2")
        assert len(filtered_ec2) == 1
        assert filtered_ec2[0].service_name == "AmazonEC2"

        # Filter by Azure account
        filtered_az = analyzer.filter_records(records, account="sub-prod-9876")
        assert len(filtered_az) == 2
        for r in filtered_az:
            assert r.provider == CloudProvider.AZURE

        # Filter by region
        filtered_reg = analyzer.filter_records(records, region="us-central1")
        assert len(filtered_reg) == 2
        for r in filtered_reg:
            assert r.region == "us-central1"

    def test_date_range_filtering(self):
        analyzer = CostAnalyzer()
        t1 = datetime.datetime(2026, 11, 5, 0, 0, tzinfo=datetime.timezone.utc)
        t2 = datetime.datetime(2026, 11, 15, 0, 0, tzinfo=datetime.timezone.utc)
        t3 = datetime.datetime(2026, 11, 25, 0, 0, tzinfo=datetime.timezone.utc)

        records = [
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=10.0,
                start_time=t1,
                end_time=t1,
            ),
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=20.0,
                start_time=t2,
                end_time=t2,
            ),
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=30.0,
                start_time=t3,
                end_time=t3,
            ),
        ]

        filtered = analyzer.filter_records(
            records,
            start_date=datetime.date(2026, 11, 10),
            end_date=datetime.date(2026, 11, 20),
        )
        assert len(filtered) == 1
        assert filtered[0].cost == 20.0

    def test_mixed_charges_and_credits_accounting(self):
        analyzer = CostAnalyzer()
        now = datetime.datetime(2026, 11, 15, 0, 0, tzinfo=datetime.timezone.utc)
        records = [
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=150.00,
                currency="USD",
                start_time=now,
                end_time=now,
            ),
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=-50.00,
                currency="USD",
                start_time=now,
                end_time=now,
            ),
        ]

        summary = analyzer.summarize(records)
        assert summary.gross_cost == 150.00
        assert summary.credit_cost == -50.00
        assert summary.total_cost == 100.00
        assert summary.service_costs["EC2"] == 100.00

    def test_zero_net_spend_and_negative_net_spend(self):
        analyzer = CostAnalyzer()
        now = datetime.datetime(2026, 11, 15, 0, 0, tzinfo=datetime.timezone.utc)
        # Net zero
        records_zero = [
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=50.00,
                currency="USD",
                start_time=now,
                end_time=now,
            ),
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=-50.00,
                currency="USD",
                start_time=now,
                end_time=now,
            ),
        ]
        summary_zero = analyzer.summarize(records_zero)
        assert summary_zero.total_cost == 0.0
        assert summary_zero.gross_cost == 50.00
        assert summary_zero.credit_cost == -50.00

        # Negative net spend (credits exceed charges)
        records_neg = [
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=20.00,
                currency="USD",
                start_time=now,
                end_time=now,
            ),
            BillingRecord(
                provider=CloudProvider.AWS,
                account_id="1",
                service_name="EC2",
                cost=-60.00,
                currency="USD",
                start_time=now,
                end_time=now,
            ),
        ]
        summary_neg = analyzer.summarize(records_neg)
        assert summary_neg.total_cost == -40.00
        assert summary_neg.gross_cost == 20.00
        assert summary_neg.credit_cost == -60.00
        # Percentage must be 0.0 when net spend is negative
        assert summary_neg.top_resources[0]["percentage"] == 0.0
