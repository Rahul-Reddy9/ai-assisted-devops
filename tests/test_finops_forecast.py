# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit and integration tests for FinOps expenditure forecasting engine (P10/P50/P90)."""

import datetime
import pytest

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.finops.forecast import CostForecaster
from devops_ai.finops.importer import get_synthetic_anomaly_and_forecast_dataset
from devops_ai.finops.models import (
    BillingRecord,
    CloudProvider,
    ForecastMethod,
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


class TestCostForecaster:
    def test_daily_forecast_horizon_and_p10_p50_p90_invariant(self):
        """Forecaster generates exact horizon points and satisfies 0 <= P10 <= P50 <= P90."""
        # 30 days of historical data with steady growth
        costs = [100.0 + i * 2.0 for i in range(30)]
        records = _make_daily_records(costs)

        forecaster = CostForecaster(
            default_horizon_days=30, confidence_level=0.90, min_observations=7
        )
        forecast = forecaster.forecast(
            records=records, horizon_days=14, method=ForecastMethod.LINEAR_REGRESSION
        )

        assert forecast.days_projected == 14
        assert len(forecast.points) == 14
        assert forecast.trend == "increasing"

        # Check quantile invariants for every point
        for pt in forecast.points:
            assert pt.lower_bound >= 0.0
            assert pt.projected_cost >= 0.0
            assert pt.upper_bound >= 0.0
            assert pt.p10 is not None and pt.p50 is not None and pt.p90 is not None
            assert 0.0 <= pt.p10 <= pt.p50 <= pt.p90
            assert pt.lower_bound == pt.p10
            assert pt.projected_cost == pt.p50
            assert pt.upper_bound == pt.p90

        # Check total sums
        assert (
            0.0
            <= forecast.total_p10_spend
            <= forecast.total_projected_spend
            <= forecast.total_p90_spend
        )

    def test_non_negative_clamping_on_declining_trend(self):
        """Even with steep downward trend, projected spend never becomes negative."""
        # Steep decline from $100 down to $10
        costs = [max(10.0, 100.0 - i * 5.0) for i in range(25)]
        records = _make_daily_records(costs)

        forecaster = CostForecaster(
            default_horizon_days=30, confidence_level=0.95, min_observations=7
        )
        forecast = forecaster.forecast(
            records=records, horizon_days=30, method=ForecastMethod.LINEAR_REGRESSION
        )

        assert forecast.trend == "decreasing"
        for pt in forecast.points:
            assert pt.lower_bound >= 0.0
            assert pt.projected_cost >= 0.0
            assert pt.upper_bound >= 0.0
            assert pt.p10 <= pt.p50 <= pt.p90

    def test_forecast_methods(self):
        """All supported forecast algorithms run cleanly and produce valid projections."""
        costs = [50.0 + (i % 5) * 2.0 for i in range(30)]
        records = _make_daily_records(costs)
        forecaster = CostForecaster(default_horizon_days=10, min_observations=7)

        methods = [
            ForecastMethod.LINEAR_REGRESSION,
            ForecastMethod.WEIGHTED_MOVING_AVERAGE,
            ForecastMethod.EXPONENTIAL_SMOOTHING,
            ForecastMethod.HISTORICAL_AVERAGE,
        ]

        for m in methods:
            fc = forecaster.forecast(records=records, horizon_days=10, method=m)
            assert len(fc.points) == 10
            assert fc.total_projected_spend > 0.0
            assert fc.total_p10_spend <= fc.total_projected_spend <= fc.total_p90_spend
            assert fc.method == m.value

    def test_monthly_frequency_forecast_calendar_aggregation_scenarios(self):
        """Monthly frequency groups projection into calendar month periods across various horizons."""
        forecaster = CostForecaster(default_horizon_days=90, min_observations=7)

        # Scenario 1: 30-day horizon starting mid-month (2026-10-15)
        # History: 2026-09-15 to 2026-10-14 (30 days)
        # Horizon 30 days -> Oct 15 to Nov 13 -> exactly 2 calendar month points:
        # Month 1: Oct 15 to Oct 31 (17 days)
        # Month 2: Nov 01 to Nov 13 (13 days)
        records_mid_month = _make_daily_records([100.0] * 30, start_date=datetime.date(2026, 9, 15))
        fc_mid = forecaster.forecast(
            records=records_mid_month, horizon_days=30, frequency="monthly"
        )
        assert len(fc_mid.points) == 2
        assert fc_mid.days_projected == 30
        assert fc_mid.points[0].date == datetime.date(2026, 10, 15)
        assert fc_mid.points[1].date == datetime.date(2026, 11, 1)
        # Oct has 17 days * $100 = $1700, Nov has 13 days * $100 = $1300
        assert fc_mid.points[0].projected_cost == pytest.approx(1700.0, rel=0.05)
        assert fc_mid.points[1].projected_cost == pytest.approx(1300.0, rel=0.05)
        assert sum(p.projected_cost for p in fc_mid.points) == pytest.approx(
            fc_mid.total_projected_spend, rel=1e-5
        )
        assert sum(p.p10 for p in fc_mid.points) == pytest.approx(fc_mid.total_p10_spend, rel=1e-5)
        assert sum(p.p90 for p in fc_mid.points) == pytest.approx(fc_mid.total_p90_spend, rel=1e-5)
        for pt in fc_mid.points:
            assert 0.0 <= pt.p10 <= pt.p50 <= pt.p90

        # Scenario 2: 90-day horizon starting month-start (2026-11-01)
        # History: 2026-10-01 to 2026-10-31 (31 days)
        # Horizon 90 days -> Nov 01 to Jan 29 -> exactly 3 calendar month points:
        # Nov: 30 days, Dec: 31 days, Jan: 29 days
        records_month_start = _make_daily_records(
            [80.0] * 31, start_date=datetime.date(2026, 10, 1)
        )
        fc_start = forecaster.forecast(
            records=records_month_start, horizon_days=90, frequency="monthly"
        )
        assert len(fc_start.points) == 3
        assert fc_start.days_projected == 90
        assert fc_start.points[0].date == datetime.date(2026, 11, 1)
        assert fc_start.points[1].date == datetime.date(2026, 12, 1)
        assert fc_start.points[2].date == datetime.date(2027, 1, 1)
        assert fc_start.points[0].projected_cost == pytest.approx(30 * 80.0, rel=0.05)
        assert fc_start.points[1].projected_cost == pytest.approx(31 * 80.0, rel=0.05)
        assert fc_start.points[2].projected_cost == pytest.approx(29 * 80.0, rel=0.05)
        for pt in fc_start.points:
            assert 0.0 <= pt.p10 <= pt.p50 <= pt.p90

        # Scenario 3: 90-day horizon crossing year boundary from mid-month
        # History: 45 days starting 2026-10-01 -> ends 2026-11-14
        # Horizon 90 days -> Nov 15 to Feb 12 -> exactly 4 calendar month points:
        # Nov: 16 days, Dec: 31 days, Jan: 31 days, Feb: 12 days
        records_cross_year = _make_daily_records([50.0] * 45, start_date=datetime.date(2026, 10, 1))
        fc_cross = forecaster.forecast(
            records=records_cross_year, horizon_days=90, frequency="monthly"
        )
        assert len(fc_cross.points) == 4
        assert fc_cross.days_projected == 90
        assert fc_cross.points[0].date == datetime.date(2026, 11, 15)
        assert fc_cross.points[1].date == datetime.date(2026, 12, 1)
        assert fc_cross.points[2].date == datetime.date(2027, 1, 1)
        assert fc_cross.points[3].date == datetime.date(2027, 2, 1)
        assert fc_cross.points[0].projected_cost == pytest.approx(16 * 50.0, rel=0.05)
        assert fc_cross.points[1].projected_cost == pytest.approx(31 * 50.0, rel=0.05)
        assert fc_cross.points[2].projected_cost == pytest.approx(31 * 50.0, rel=0.05)
        assert fc_cross.points[3].projected_cost == pytest.approx(12 * 50.0, rel=0.05)
        for pt in fc_cross.points:
            assert 0.0 <= pt.p10 <= pt.p50 <= pt.p90

        # Verify comonotonicity assumption documentation in supporting evidence
        assert any("comonotonicity" in ev.lower() for ev in fc_mid.supporting_evidence)

    def test_in_sample_goodness_of_fit_metrics(self):
        """In-sample evaluation metrics (MAE, RMSE, MAPE) measure historical fit."""
        costs = [100.0, 102.0, 98.0, 105.0, 101.0, 99.0, 103.0, 100.0]
        records = _make_daily_records(costs)

        forecaster = CostForecaster(min_observations=5)
        fc = forecaster.forecast(records=records, horizon_days=7)

        assert "mae" in fc.evaluation_metrics
        assert "rmse" in fc.evaluation_metrics
        assert "mape" in fc.evaluation_metrics
        assert fc.evaluation_metrics["mae"] >= 0.0
        assert fc.evaluation_metrics["rmse"] >= 0.0
        assert fc.evaluation_metrics["mape"] >= 0.0
        # Verify terminology in supporting evidence is "in-sample goodness-of-fit"
        assert any("in-sample goodness-of-fit" in ev.lower() for ev in fc.supporting_evidence)
        assert not any("backtest accuracy" in ev.lower() for ev in fc.supporting_evidence)

    def test_gaussian_quantile_approximation_caveat_documented(self):
        """Forecast supporting evidence documents the zero-clamped Gaussian approximation."""
        costs = [50.0 + (i % 4) * 5.0 for i in range(20)]
        records = _make_daily_records(costs)
        forecaster = CostForecaster(min_observations=7)
        fc = forecaster.forecast(records=records, horizon_days=14)

        assert any("gaussian" in ev.lower() for ev in fc.supporting_evidence)
        assert fc.evaluation_metrics["mape"] >= 0.0

    def test_insufficient_historical_data_handling(self):
        """Forecaster handles series with fewer observations than min_observations."""
        costs = [100.0, 110.0]  # Only 2 points, min is 7
        records = _make_daily_records(costs)

        forecaster = CostForecaster(default_horizon_days=14, min_observations=7)
        fc = forecaster.forecast(records=records, horizon_days=14)

        assert fc.insufficient_data is True
        assert fc.message is not None
        assert len(fc.points) == 0
        assert fc.total_projected_spend == 0.0

    def test_currency_mismatch_raises_error(self):
        """Forecast fails if multiple currencies are present in billing records."""
        records = _make_daily_records([100.0] * 10, currency="USD")
        records.extend(_make_daily_records([100.0] * 5, currency="GBP"))

        forecaster = CostForecaster()
        with pytest.raises(CurrencyMismatchError):
            forecaster.forecast(records=records)

    def test_service_filter(self):
        """Service filter isolates spend to matching service."""
        records_ec2 = _make_daily_records([100.0] * 20, service="AmazonEC2")
        records_s3 = _make_daily_records([25.0] * 20, service="AmazonS3")
        records = records_ec2 + records_s3

        forecaster = CostForecaster(default_horizon_days=10, min_observations=7)
        fc = forecaster.forecast(records=records, horizon_days=10, service_filter="AmazonS3")

        # Daily baseline should be ~25.0
        daily_avg = fc.total_projected_spend / 10.0
        assert daily_avg == pytest.approx(25.0, rel=0.1)

    def test_summarize_forecast(self):
        """summarize_forecast creates valid ForecastSummary."""
        costs = [50.0] * 15
        records = _make_daily_records(costs)

        forecaster = CostForecaster(default_horizon_days=10, min_observations=7)
        fc = forecaster.forecast(records=records, horizon_days=10)
        summary = forecaster.summarize_forecast(fc)

        assert summary.horizon_days == 10
        assert summary.total_projected_cost == pytest.approx(500.0, rel=0.1)
        assert summary.daily_average_cost == pytest.approx(50.0, rel=0.1)
        assert summary.total_p10_cost <= summary.total_projected_cost <= summary.total_p90_cost
        assert summary.forecast_start_date < summary.forecast_end_date

    def test_synthetic_benchmark_forecast(self):
        """Forecasting over synthetic benchmark produces realistic projections."""
        records = get_synthetic_anomaly_and_forecast_dataset()
        forecaster = CostForecaster(
            default_horizon_days=30, confidence_level=0.90, min_observations=7
        )
        fc = forecaster.forecast(records=records, horizon_days=30)

        assert fc.days_projected == 30
        assert len(fc.points) == 30
        assert fc.trend in ["increasing", "stable"]
        assert fc.total_projected_spend > 5000.0
        assert fc.total_p10_spend <= fc.total_projected_spend <= fc.total_p90_spend
        assert len(fc.supporting_evidence) >= 1
