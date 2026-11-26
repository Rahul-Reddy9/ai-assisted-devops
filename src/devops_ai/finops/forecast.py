# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Expenditure forecasting engine with uncertainty-aware prediction intervals (P10/P50/P90)."""

from collections import defaultdict
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.core.logging import get_logger
from devops_ai.finops.models import (
    BillingRecord,
    CostForecast,
    ForecastMethod,
    ForecastPoint,
    ForecastSummary,
)

logger = get_logger("finops.forecast")


class CostForecaster:
    """Generates explainable, uncertainty-bounded cloud expenditure projections."""

    def __init__(
        self,
        default_horizon_days: int = 30,
        confidence_level: float = 0.90,
        min_observations: int = 7,
    ) -> None:
        """Initialize forecaster with projection parameters.

        Args:
            default_horizon_days: Default future window in days for projections.
            confidence_level: Confidence level for prediction intervals (0.50 - 0.99).
            min_observations: Minimum historical daily points required for reliable forecast.
        """
        self.default_horizon_days = int(default_horizon_days)
        self.confidence_level = float(confidence_level)
        self.min_observations = int(min_observations)

    @staticmethod
    def _get_z_multiplier(confidence_level: float) -> float:
        """Return normal distribution critical multiplier for given confidence level."""
        # Standard two-sided z-critical values
        if confidence_level >= 0.99:
            return 2.576
        if confidence_level >= 0.95:
            return 1.960
        if confidence_level >= 0.90:
            return 1.645
        if confidence_level >= 0.80:
            return 1.282
        return 1.000

    def forecast(
        self,
        records: List[BillingRecord],
        horizon_days: Optional[int] = None,
        method: Optional[ForecastMethod] = None,
        base_currency: Optional[str] = None,
        service_filter: Optional[str] = None,
        account_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
        frequency: str = "daily",
    ) -> CostForecast:
        """Generate forward-looking expenditure forecast from historical billing records.

        Args:
            records: Historical normalized billing records.
            horizon_days: Number of days into future to project (defaults to config default).
            method: Forecasting algorithm to apply. Defaults to linear regression / Holt trend.
            base_currency: Expected 3-letter currency code (e.g. 'USD').
            service_filter: Filter history to a specific cloud service.
            account_filter: Filter history to a specific cloud account.
            region_filter: Filter history to a specific cloud region.
            frequency: Granularity of projection ('daily' or 'monthly').

        Returns:
            CostForecast instance containing projected points, quantiles (P10/P50/P90), and evidence.

        Raises:
            CurrencyMismatchError: If records contain mixed, incompatible currencies.
        """
        horizon = horizon_days if horizon_days is not None else self.default_horizon_days
        if horizon < 1:
            horizon = self.default_horizon_days

        target_method = method or ForecastMethod.LINEAR_REGRESSION

        if not records:
            return CostForecast(
                days_projected=horizon,
                method=target_method.value,
                total_projected_spend=0.0,
                total_p10_spend=0.0,
                total_p90_spend=0.0,
                currency=base_currency or "USD",
                points=[],
                evaluation_metrics={"mae": 0.0, "rmse": 0.0, "mape": 0.0},
                supporting_evidence=["No historical billing records provided for analysis."],
                insufficient_data=True,
                message="Cannot generate forecast: billing dataset is empty.",
            )

        # 1. Enforce strict currency consistency
        target_currency = (base_currency or records[0].currency).strip().upper()
        for idx, r in enumerate(records):
            r_curr = r.currency.strip().upper()
            if r_curr != target_currency:
                logger.error(
                    f"Record {idx} ({r.record_id}) currency '{r_curr}' does not match expected '{target_currency}'"
                )
                raise CurrencyMismatchError(base_currency=target_currency, found_currency=r_curr)

        # 2. Filter records by dimensional criteria
        filtered_records = records
        dim_target = "total"
        if service_filter:
            s_clean = service_filter.strip().lower()
            filtered_records = [r for r in filtered_records if s_clean in r.service_name.lower()]
            dim_target = f"service:{service_filter}"
        if account_filter:
            a_clean = account_filter.strip().lower()
            filtered_records = [r for r in filtered_records if a_clean in r.account_id.lower()]
            dim_target = f"account:{account_filter}"
        if region_filter:
            reg_clean = region_filter.strip().lower()
            filtered_records = [
                r for r in filtered_records if r.region and reg_clean in r.region.lower()
            ]
            dim_target = f"region:{region_filter}"

        # 3. Aggregate daily time series
        daily_spend: Dict[datetime.date, float] = defaultdict(float)
        for r in filtered_records:
            d = r.start_time.date()
            daily_spend[d] += float(r.cost)

        if not daily_spend:
            return CostForecast(
                target_dimension=dim_target,
                days_projected=horizon,
                method=target_method.value,
                total_projected_spend=0.0,
                total_p10_spend=0.0,
                total_p90_spend=0.0,
                currency=target_currency,
                points=[],
                evaluation_metrics={"mae": 0.0, "rmse": 0.0, "mape": 0.0},
                supporting_evidence=[
                    "No matching historical spend found for the requested filters."
                ],
                insufficient_data=True,
                message="Cannot generate forecast: no expenditure records match the selected filters.",
            )

        # Fill chronological date series from earliest date to latest date
        sorted_dates = sorted(daily_spend.keys())
        first_date = sorted_dates[0]
        last_date = sorted_dates[-1]

        dense_history: List[Tuple[datetime.date, float]] = []
        curr = first_date
        while curr <= last_date:
            dense_history.append((curr, daily_spend.get(curr, 0.0)))
            curr += datetime.timedelta(days=1)

        n_obs = len(dense_history)

        # Check minimum observation requirement
        if n_obs < self.min_observations:
            return CostForecast(
                target_dimension=dim_target,
                days_projected=horizon,
                method=target_method.value,
                total_projected_spend=0.0,
                total_p10_spend=0.0,
                total_p90_spend=0.0,
                currency=target_currency,
                points=[],
                evaluation_metrics={"mae": 0.0, "rmse": 0.0, "mape": 0.0},
                supporting_evidence=[
                    f"Available observation window has only {n_obs} daily data points.",
                    f"A minimum of {self.min_observations} observation points is required for reliable statistical forecasting.",
                ],
                insufficient_data=True,
                message=f"Insufficient historical data: available {n_obs} days < required {self.min_observations} days.",
            )

        # 4. Fit statistical forecasting model
        y_vals = [max(0.0, s) for _, s in dense_history]
        x_vals = list(range(n_obs))

        # Model fitting
        if target_method == ForecastMethod.LINEAR_REGRESSION:
            fitted_vals, forecast_fn, se = self._fit_linear_regression(x_vals, y_vals)
        elif target_method == ForecastMethod.WEIGHTED_MOVING_AVERAGE:
            fitted_vals, forecast_fn, se = self._fit_weighted_moving_average(y_vals)
        elif target_method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            fitted_vals, forecast_fn, se = self._fit_exponential_smoothing(y_vals)
        else:
            # Historical average baseline
            fitted_vals, forecast_fn, se = self._fit_historical_average(y_vals)

        # 5. Compute backtesting evaluation metrics
        mae, rmse, mape = self._compute_evaluation_metrics(y_vals, fitted_vals)

        # 6. Generate forecast points with uncertainty bounds
        z_multiplier = self._get_z_multiplier(self.confidence_level)
        z_p10_p90 = 1.282  # 80% coverage interval for P10-P90 percentiles

        points: List[ForecastPoint] = []
        total_p50 = 0.0
        total_p10 = 0.0
        total_p90 = 0.0

        for h in range(1, horizon + 1):
            future_date = last_date + datetime.timedelta(days=h)
            p50_val = max(0.0, forecast_fn(h))

            # Variance expands with horizon distance
            horizon_penalty = math.sqrt(1.0 + (h / float(n_obs)))
            std_err_h = se * horizon_penalty

            p10_val = max(0.0, p50_val - (z_p10_p90 * std_err_h))
            p90_val = max(p50_val, p50_val + (z_p10_p90 * std_err_h))

            # Prediction interval bounds based on requested confidence level
            lower_bound = max(0.0, p50_val - (z_multiplier * std_err_h))
            upper_bound = max(p50_val, p50_val + (z_multiplier * std_err_h))

            # Enforce non-negative ordering invariant: 0 <= p10 <= p50 <= p90
            p10_clean = round(p10_val, 2)
            p50_clean = round(p50_val, 2)
            p90_clean = round(p90_val, 2)

            pt = ForecastPoint(
                date=future_date,
                projected_cost=p50_clean,
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2),
                p10=p10_clean,
                p50=p50_clean,
                p90=p90_clean,
            )
            points.append(pt)
            total_p50 += p50_clean
            total_p10 += p10_clean
            total_p90 += p90_clean

        # Convert daily points to monthly points if requested (F-01)
        norm_freq = frequency.strip().lower()
        monthly_buckets: Dict[Tuple[int, int], List[ForecastPoint]] = defaultdict(list)
        if norm_freq == "monthly" and points:
            for pt in points:
                monthly_buckets[(pt.date.year, pt.date.month)].append(pt)

            monthly_points: List[ForecastPoint] = []
            for (year, month), m_pts in monthly_buckets.items():
                m_start_date = m_pts[0].date
                m_p50 = sum(p.p50 for p in m_pts if p.p50 is not None)
                m_p10 = sum(p.p10 for p in m_pts if p.p10 is not None)
                m_p90 = sum(p.p90 for p in m_pts if p.p90 is not None)
                m_lower = sum(p.lower_bound for p in m_pts)
                m_upper = sum(p.upper_bound for p in m_pts)

                m_p10_clean = round(max(0.0, m_p10), 2)
                m_p50_clean = round(max(m_p10_clean, m_p50), 2)
                m_p90_clean = round(max(m_p50_clean, m_p90), 2)

                monthly_points.append(
                    ForecastPoint(
                        date=m_start_date,
                        projected_cost=m_p50_clean,
                        lower_bound=round(max(0.0, m_lower), 2),
                        upper_bound=round(max(m_p50_clean, m_upper), 2),
                        p10=m_p10_clean,
                        p50=m_p50_clean,
                        p90=m_p90_clean,
                    )
                )

            points = monthly_points
            total_p50 = sum(p.projected_cost for p in points)
            total_p10 = sum(p.p10 for p in points if p.p10 is not None)
            total_p90 = sum(p.p90 for p in points if p.p90 is not None)

        # 7. Determine overall trajectory trend
        if target_method == ForecastMethod.LINEAR_REGRESSION:
            mean_x = sum(x_vals) / float(n_obs)
            mean_y = sum(y_vals) / float(n_obs)
            denom = sum((xi - mean_x) ** 2 for xi in x_vals)
            slope = (
                sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n_obs)) / denom
                if denom > 0.0
                else 0.0
            )
            rel_slope = (slope / mean_y * 100.0) if mean_y > 0.0 else (slope * 100.0)
            if rel_slope >= 1.0 or slope >= 0.5:
                trend = "increasing"
            elif rel_slope <= -1.0 or slope <= -0.5:
                trend = "decreasing"
            else:
                trend = "stable"
        elif len(points) >= 2:
            first_p50 = points[0].projected_cost
            last_p50 = points[-1].projected_cost
            if first_p50 > 0.0:
                pct_change = (last_p50 - first_p50) / first_p50 * 100.0
            else:
                pct_change = 100.0 if last_p50 > 0 else 0.0

            if pct_change >= 3.0:
                trend = "increasing"
            elif pct_change <= -3.0:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # 8. Compile supporting evidence and methodological caveats
        evidence = [
            f"Fitted model '{target_method.value}' across {n_obs} historical daily observations ({first_date.isoformat()} to {last_date.isoformat()}).",
            f"Model in-sample goodness-of-fit: MAE = ${mae:.2f} {target_currency}, RMSE = ${rmse:.2f} {target_currency}, MAPE = {mape:.1f}%.",
            f"Residual standard error (SE): ${se:.2f} {target_currency} per day.",
            f"Projected {len(points)} {'months' if norm_freq == 'monthly' else 'days'} ({points[0].date.isoformat()} to {points[-1].date.isoformat()}) displaying an '{trend}' trajectory.",
            f"Confidence intervals calculated at {self.confidence_level:.0%} coverage with expanding horizon uncertainty.",
            "Prediction intervals (P10/P50/P90) use zero-clamped Gaussian approximations assuming persistent daily variance; they indicate statistical variance from historical patterns and are not financial guarantees.",
        ]
        if norm_freq == "monthly" and monthly_buckets:
            month_spans = [
                f"{p.date.strftime('%Y-%m')} ({len(monthly_buckets[(p.date.year, p.date.month)])}d)"
                for p in points
            ]
            evidence.append(
                f"Monthly frequency aggregated {horizon} days into {len(points)} calendar-month periods: {', '.join(month_spans)}."
            )
            evidence.append(
                "Monthly prediction intervals (P10/P90) assume positive dependence across daily forecast errors (comonotonicity), providing a prudent upper bound on monthly variance."
            )

        return CostForecast(
            target_dimension=dim_target,
            days_projected=horizon,
            method=target_method.value,
            total_projected_spend=round(total_p50, 2),
            total_p10_spend=round(total_p10, 2),
            total_p90_spend=round(total_p90, 2),
            currency=target_currency,
            confidence_level=self.confidence_level,
            historical_data_points=n_obs,
            trend=trend,
            points=points,
            evaluation_metrics={
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 1),
            },
            supporting_evidence=evidence,
            insufficient_data=False,
            message=None,
        )

    def summarize_forecast(self, forecast: CostForecast) -> ForecastSummary:
        """Convert a CostForecast into a structured ForecastSummary executive report."""
        if not forecast.points:
            today = datetime.date.today()
            return ForecastSummary(
                forecast_id=forecast.forecast_id,
                currency=forecast.currency,
                horizon_days=forecast.days_projected,
                forecast_start_date=today,
                forecast_end_date=today + datetime.timedelta(days=forecast.days_projected),
                total_projected_cost=0.0,
                total_p10_cost=0.0,
                total_p90_cost=0.0,
                daily_average_cost=0.0,
                trend="stable",
                method=forecast.method,
                confidence_level=forecast.confidence_level,
                historical_days_analyzed=forecast.historical_data_points,
                forecast=forecast,
            )

        start_d = forecast.points[0].date
        end_d = forecast.points[-1].date
        n_days = len(forecast.points)
        daily_avg = forecast.total_projected_spend / float(n_days) if n_days > 0 else 0.0

        return ForecastSummary(
            forecast_id=forecast.forecast_id,
            currency=forecast.currency,
            horizon_days=n_days,
            forecast_start_date=start_d,
            forecast_end_date=end_d,
            total_projected_cost=forecast.total_projected_spend,
            total_p10_cost=forecast.total_p10_spend,
            total_p90_cost=forecast.total_p90_spend,
            daily_average_cost=round(daily_avg, 2),
            trend=forecast.trend,
            method=forecast.method,
            confidence_level=forecast.confidence_level,
            historical_days_analyzed=forecast.historical_data_points,
            forecast=forecast,
        )

    # -----------------------------------------------------------------
    # Statistical Fitting Primitives
    # -----------------------------------------------------------------

    @staticmethod
    def _fit_linear_regression(x: List[int], y: List[float]) -> Tuple[List[float], Any, float]:
        """Fit ordinary least squares linear regression."""
        n = len(x)
        mean_x = sum(x) / float(n)
        mean_y = sum(y) / float(n)

        denom = sum((xi - mean_x) ** 2 for xi in x)
        if denom > 0.0:
            slope = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / denom
            intercept = mean_y - (slope * mean_x)
        else:
            slope = 0.0
            intercept = mean_y

        fitted = [intercept + (slope * xi) for xi in x]

        # Residual standard error
        residuals = [y[i] - fitted[i] for i in range(n)]
        dof = max(1, n - 2)
        se = math.sqrt(sum(r**2 for r in residuals) / float(dof))

        def forecast_fn(h: int) -> float:
            target_x = (n - 1) + h
            return intercept + (slope * target_x)

        return fitted, forecast_fn, se

    @staticmethod
    def _fit_weighted_moving_average(
        y: List[float], window_size: int = 14
    ) -> Tuple[List[float], Any, float]:
        """Fit linearly weighted moving average."""
        n = len(y)
        k = min(n, window_size)
        weights = list(range(1, k + 1))
        weight_sum = sum(weights)

        recent_slice = y[-k:]
        wma_val = sum(recent_slice[i] * weights[i] for i in range(k)) / float(weight_sum)

        # Baseline fitted values
        fitted: List[float] = []
        for i in range(n):
            if i < 2:
                fitted.append(y[i])
            else:
                sub_k = min(i, k)
                sub_w = list(range(1, sub_k + 1))
                sub_sum = sum(sub_w)
                sub_slice = y[i - sub_k : i]
                fitted.append(sum(sub_slice[j] * sub_w[j] for j in range(sub_k)) / float(sub_sum))

        residuals = [y[i] - fitted[i] for i in range(n)]
        dof = max(1, n - 1)
        se = math.sqrt(sum(r**2 for r in residuals) / float(dof))

        def forecast_fn(h: int) -> float:
            return wma_val

        return fitted, forecast_fn, se

    @staticmethod
    def _fit_exponential_smoothing(
        y: List[float], alpha: float = 0.3, beta: float = 0.1
    ) -> Tuple[List[float], Any, float]:
        """Fit Holt's two-parameter linear exponential smoothing."""
        n = len(y)
        level = y[0]
        trend = (y[-1] - y[0]) / float(n) if n > 1 else 0.0

        fitted: List[float] = [level]
        for i in range(1, n):
            val = y[i]
            last_level = level
            level = alpha * val + (1.0 - alpha) * (last_level + trend)
            trend = beta * (level - last_level) + (1.0 - beta) * trend
            fitted.append(last_level + trend)

        residuals = [y[i] - fitted[i] for i in range(n)]
        dof = max(1, n - 2)
        se = math.sqrt(sum(r**2 for r in residuals) / float(dof))

        final_level = level
        final_trend = trend

        def forecast_fn(h: int) -> float:
            return final_level + (h * final_trend)

        return fitted, forecast_fn, se

    @staticmethod
    def _fit_historical_average(y: List[float]) -> Tuple[List[float], Any, float]:
        """Fit simple historical mean."""
        n = len(y)
        mean_val = sum(y) / float(n) if n > 0 else 0.0
        fitted = [mean_val] * n

        residuals = [yi - mean_val for yi in y]
        dof = max(1, n - 1)
        se = math.sqrt(sum(r**2 for r in residuals) / float(dof))

        def forecast_fn(h: int) -> float:
            return mean_val

        return fitted, forecast_fn, se

    @staticmethod
    def _compute_evaluation_metrics(
        actual: List[float], fitted: List[float]
    ) -> Tuple[float, float, float]:
        """Calculate in-sample MAE, RMSE, and MAPE."""
        n = len(actual)
        if n == 0:
            return 0.0, 0.0, 0.0

        errors = [abs(actual[i] - fitted[i]) for i in range(n)]
        mae = sum(errors) / float(n)
        rmse = math.sqrt(sum((actual[i] - fitted[i]) ** 2 for i in range(n)) / float(n))

        pct_errors = [(errors[i] / actual[i] * 100.0) for i in range(n) if actual[i] > 0.001]
        mape = (sum(pct_errors) / float(len(pct_errors))) if pct_errors else 0.0

        return mae, rmse, mape
