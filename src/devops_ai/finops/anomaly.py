# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Statistical cost anomaly detection engine across multi-dimensional cloud billing data."""

from collections import defaultdict
import datetime
import math
from typing import Dict, List, Optional, Set, Tuple

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.core.logging import get_logger
from devops_ai.finops.models import (
    AnomalySeverity,
    AnomalySummary,
    BillingRecord,
    CloudProvider,
    CostAnomaly,
)

logger = get_logger("finops.anomaly")


class CostAnomalyDetector:
    """Detects and classifies unexpected expenditure surges and anomalies across cloud dimensions."""

    def __init__(
        self,
        z_score_threshold: float = 2.5,
        min_observations: int = 7,
        rolling_window_days: int = 30,
        min_absolute_delta: float = 10.0,
        min_percentage_increase: float = 20.0,
    ) -> None:
        """Initialize detector with configurable sensitivity thresholds.

        Args:
            z_score_threshold: Minimum standard deviations from baseline to flag an anomaly.
            min_observations: Minimum historical observation days required for baseline.
            rolling_window_days: Maximum size of historical rolling window.
            min_absolute_delta: Minimum monetary spend increase to avoid flagging micro-variances.
            min_percentage_increase: Minimum percentage increase above baseline to consider.
        """
        self.z_score_threshold = float(z_score_threshold)
        self.min_observations = int(min_observations)
        self.rolling_window_days = int(rolling_window_days)
        self.min_absolute_delta = float(min_absolute_delta)
        self.min_percentage_increase = float(min_percentage_increase)

    @staticmethod
    def _calculate_mean_and_stdev(values: List[float]) -> Tuple[float, float]:
        """Compute sample mean and sample standard deviation."""
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mean = sum(values) / n
        if n < 2:
            return mean, 0.0
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        return mean, math.sqrt(max(0.0, variance))

    @staticmethod
    def _calculate_median_and_mad(values: List[float]) -> Tuple[float, float]:
        """Compute median and Median Absolute Deviation (MAD)."""
        if not values:
            return 0.0, 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        median = sorted_vals[mid] if n % 2 != 0 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        deviations = sorted(abs(x - median) for x in sorted_vals)
        mad = deviations[mid] if n % 2 != 0 else (deviations[mid - 1] + deviations[mid]) / 2.0
        return median, mad

    def detect_anomalies(
        self,
        records: List[BillingRecord],
        base_currency: Optional[str] = None,
        service_filter: Optional[str] = None,
        account_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
        severity_filter: Optional[AnomalySeverity] = None,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        fill_missing_dates: bool = True,
    ) -> AnomalySummary:
        """Scan historical billing records and extract statistical expenditure anomalies.

        Args:
            records: List of normalized BillingRecord instances.
            base_currency: Expected 3-letter currency code (e.g. 'USD').
            service_filter: Restrict anomaly search to matching cloud service.
            account_filter: Restrict anomaly search to matching account ID.
            region_filter: Restrict anomaly search to matching cloud region.
            severity_filter: Minimum anomaly severity level to include in summary.
            start_date: Optional lower bound for anomaly observation window.
            end_date: Optional upper bound for anomaly observation window.
            fill_missing_dates: Whether days without records in range are treated as zero spend.

        Returns:
            AnomalySummary containing detected anomalies and aggregated metrics.

        Raises:
            CurrencyMismatchError: If records contain mixed, incompatible currencies.
        """
        if not records:
            return AnomalySummary(
                total_anomalies=0,
                total_excess_spend=0.0,
                currency=base_currency or "USD",
                anomalies_by_severity={},
                anomalies_by_service={},
                period_start=start_date,
                period_end=end_date,
                anomalies=[],
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
        if service_filter:
            s_clean = service_filter.strip().lower()
            filtered_records = [r for r in filtered_records if s_clean in r.service_name.lower()]
        if account_filter:
            a_clean = account_filter.strip().lower()
            filtered_records = [r for r in filtered_records if a_clean in r.account_id.lower()]
        if region_filter:
            reg_clean = region_filter.strip().lower()
            filtered_records = [
                r for r in filtered_records if r.region and reg_clean in r.region.lower()
            ]

        if not filtered_records:
            return AnomalySummary(
                total_anomalies=0,
                total_excess_spend=0.0,
                currency=target_currency,
                anomalies_by_severity={},
                anomalies_by_service={},
                period_start=start_date,
                period_end=end_date,
                anomalies=[],
            )

        # 3. Build multi-dimensional time series
        # Structure: dim_key -> date -> daily_cost
        overall_daily: Dict[datetime.date, float] = defaultdict(float)
        service_daily: Dict[str, Dict[datetime.date, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        account_daily: Dict[str, Dict[datetime.date, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        region_daily: Dict[str, Dict[datetime.date, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Detailed attribution: (dim_key, date) -> list of records
        daily_details: Dict[Tuple[str, datetime.date], List[BillingRecord]] = defaultdict(list)

        all_dates: Set[datetime.date] = set()

        for r in filtered_records:
            d = r.start_time.date()
            all_dates.add(d)
            cost = float(r.cost)

            overall_daily[d] += cost
            daily_details[("overall", d)].append(r)

            service_daily[r.service_name][d] += cost
            daily_details[(f"service:{r.service_name}", d)].append(r)

            account_daily[r.account_id][d] += cost
            daily_details[(f"account:{r.account_id}", d)].append(r)

            if r.region:
                region_daily[r.region][d] += cost
                daily_details[(f"region:{r.region}", d)].append(r)

        sorted_all_dates = sorted(all_dates)
        first_date = sorted_all_dates[0]
        last_date = sorted_all_dates[-1]

        # 4. Analyze dimensions for anomalies
        detected_anomalies: List[CostAnomaly] = []

        # Analyze overall series (only when no dimensional filter is set to prevent duplicate overall series)
        is_filtered = bool(service_filter or account_filter or region_filter)
        if not is_filtered:
            self._analyze_series(
                series_name="overall",
                daily_data=overall_daily,
                first_date=first_date,
                last_date=last_date,
                fill_missing=fill_missing_dates,
                currency=target_currency,
                daily_details=daily_details,
                detected=detected_anomalies,
            )

        # Analyze each service series (F-03: start from each service's actual start date)
        for svc_name, svc_series in service_daily.items():
            svc_first_date = min(svc_series.keys()) if svc_series else first_date
            self._analyze_series(
                series_name=f"service:{svc_name}",
                daily_data=svc_series,
                first_date=svc_first_date,
                last_date=last_date,
                fill_missing=fill_missing_dates,
                currency=target_currency,
                daily_details=daily_details,
                detected=detected_anomalies,
                service_name=svc_name,
            )

        # Analyze each account series (F-02 & F-03)
        for acct_id, acct_series in account_daily.items():
            acct_first_date = min(acct_series.keys()) if acct_series else first_date
            self._analyze_series(
                series_name=f"account:{acct_id}",
                daily_data=acct_series,
                first_date=acct_first_date,
                last_date=last_date,
                fill_missing=fill_missing_dates,
                currency=target_currency,
                daily_details=daily_details,
                detected=detected_anomalies,
                account_id=acct_id,
            )

        # Analyze each region series (F-02 & F-03)
        for reg_name, reg_series in region_daily.items():
            reg_first_date = min(reg_series.keys()) if reg_series else first_date
            self._analyze_series(
                series_name=f"region:{reg_name}",
                daily_data=reg_series,
                first_date=reg_first_date,
                last_date=last_date,
                fill_missing=fill_missing_dates,
                currency=target_currency,
                daily_details=daily_details,
                detected=detected_anomalies,
                region=reg_name,
            )

        # Deduplicate redundant dimensional anomalies across overlapping hierarchies
        # Specificity order: service (4) > account (3) > region (2) > overall (1)
        def _specificity(anom: CostAnomaly) -> int:
            if anom.service_name:
                return 4
            if anom.account_id:
                return 3
            if anom.region:
                return 2
            return 1

        def _same_entity(a: CostAnomaly, b: CostAnomaly) -> bool:
            """Check if two anomalies refer to the exact same dimensional entity."""
            if a.service_name or b.service_name:
                return a.service_name == b.service_name
            if a.account_id or b.account_id:
                return a.account_id == b.account_id
            if a.region or b.region:
                return a.region == b.region
            return True

        detected_anomalies.sort(key=lambda a: (a.cost_delta, _specificity(a)), reverse=True)

        deduped_anomalies: List[CostAnomaly] = []
        for anom in detected_anomalies:
            is_dup = False
            for existing in deduped_anomalies:
                if existing.detected_date == anom.detected_date:
                    delta_diff = abs(existing.cost_delta - anom.cost_delta)
                    cost_diff = abs(existing.actual_cost - anom.actual_cost)
                    if delta_diff < 0.10:
                        spec_existing = _specificity(existing)
                        spec_anom = _specificity(anom)
                        if spec_existing > spec_anom:
                            # Higher-level aggregate is redundant shadow of more specific root cause
                            is_dup = True
                            break
                        elif (
                            spec_existing == spec_anom
                            and _same_entity(existing, anom)
                            and cost_diff < 0.10
                        ):
                            # Duplicate report for the exact same entity on the same date
                            is_dup = True
                            break
            if not is_dup:
                deduped_anomalies.append(anom)

        # 5. Apply temporal and severity filters
        severity_rank = {
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.CRITICAL: 4,
        }
        min_severity_level = severity_rank[severity_filter] if severity_filter else 1

        final_anomalies: List[CostAnomaly] = []
        for anom in deduped_anomalies:
            if start_date and anom.detected_date < start_date:
                continue
            if end_date and anom.detected_date > end_date:
                continue
            if severity_rank[anom.severity] < min_severity_level:
                continue
            final_anomalies.append(anom)

        # Sort anomalies by cost_delta descending
        final_anomalies.sort(key=lambda a: a.cost_delta, reverse=True)

        # 6. Aggregate summary metrics
        total_excess = sum(a.cost_delta for a in final_anomalies)
        severity_counts: Dict[str, int] = defaultdict(int)
        service_counts: Dict[str, int] = defaultdict(int)

        for a in final_anomalies:
            severity_counts[a.severity.value] += 1
            if a.service_name:
                service_counts[a.service_name] += 1
            elif a.account_id:
                service_counts[f"account:{a.account_id}"] += 1
            elif a.region:
                service_counts[f"region:{a.region}"] += 1
            else:
                service_counts["overall"] += 1

        return AnomalySummary(
            total_anomalies=len(final_anomalies),
            total_excess_spend=round(total_excess, 2),
            currency=target_currency,
            anomalies_by_severity=dict(severity_counts),
            anomalies_by_service=dict(service_counts),
            period_start=start_date or first_date,
            period_end=end_date or last_date,
            anomalies=final_anomalies,
        )

    def _analyze_series(
        self,
        series_name: str,
        daily_data: Dict[datetime.date, float],
        first_date: datetime.date,
        last_date: datetime.date,
        fill_missing: bool,
        currency: str,
        daily_details: Dict[Tuple[str, datetime.date], List[BillingRecord]],
        detected: List[CostAnomaly],
        service_name: Optional[str] = None,
        account_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        """Run statistical anomaly detection over a single chronological time series."""
        # Build dense chronological list of (date, spend)
        date_list: List[Tuple[datetime.date, float]] = []
        curr = first_date
        while curr <= last_date:
            if curr in daily_data:
                date_list.append((curr, daily_data[curr]))
            elif fill_missing:
                date_list.append((curr, 0.0))
            curr += datetime.timedelta(days=1)

        if len(date_list) < self.min_observations + 1:
            return

        for idx in range(self.min_observations, len(date_list)):
            test_date, test_spend = date_list[idx]
            if test_spend <= 0.0:
                # Do not flag zero or negative adjustments as positive spending surges
                continue

            # Historical baseline window (strictly past observations to prevent data leakage)
            start_window_idx = max(0, idx - self.rolling_window_days)
            baseline_slice = [s for _, s in date_list[start_window_idx:idx] if s >= 0.0]

            if len(baseline_slice) < self.min_observations:
                continue

            mean, stdev = self._calculate_mean_and_stdev(baseline_slice)
            median, mad = self._calculate_median_and_mad(baseline_slice)

            delta = test_spend - mean
            if delta < self.min_absolute_delta:
                continue

            pct_increase = (delta / mean * 100.0) if mean > 0.0 else 100.0
            if pct_increase < self.min_percentage_increase:
                continue

            # Calculate Z-score
            if stdev > 0.001:
                z_score = delta / stdev
            else:
                # If baseline variance is near zero, use percentage or step deviation
                z_score = 4.0 if delta >= self.min_absolute_delta * 2 else 2.5

            if z_score < self.z_score_threshold:
                continue

            # Classify severity based on standard deviations and absolute monetary impact
            if z_score >= 4.0 or (pct_increase >= 200.0 and delta >= 200.0):
                severity = AnomalySeverity.CRITICAL
            elif z_score >= 3.0 or (pct_increase >= 100.0 and delta >= 100.0):
                severity = AnomalySeverity.HIGH
            elif z_score >= 2.5 or (pct_increase >= 50.0 and delta >= 50.0):
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW

            # Calculate confidence score based on sample size and signal magnitude
            obs_bonus = min(0.20, 0.01 * (len(baseline_slice) - self.min_observations))
            z_bonus = min(0.08, 0.02 * min(4.0, z_score))
            confidence = round(min(0.99, 0.70 + obs_bonus + z_bonus), 2)

            # Identify driver resources contributing to the anomaly on this date
            rel_records = daily_details.get((series_name, test_date), [])
            driver_resources: List[str] = []
            provider_val: Optional[CloudProvider] = None

            if rel_records:
                provider_val = rel_records[0].provider
                # Group by resource_id
                res_cost: Dict[str, float] = defaultdict(float)
                for rec in rel_records:
                    if rec.resource_id:
                        res_cost[rec.resource_id] += rec.cost
                top_res = sorted(res_cost.items(), key=lambda x: x[1], reverse=True)
                driver_resources = [r[0] for r in top_res[:5]]

            evidence = [
                f"Observed expenditure of ${test_spend:,.2f} {currency} on {test_date.isoformat()} exceeded historical baseline of ${mean:,.2f} {currency}.",
                f"Statistical deviation: Z-score = {z_score:.2f}σ across {len(baseline_slice)} baseline observation days (std dev = ${stdev:,.2f}).",
                f"Variance magnitude: +{pct_increase:.1f}% (+${delta:,.2f} {currency} excess spend above baseline).",
                f"Robust median baseline was ${median:,.2f} (MAD reference = ${mad:,.2f}) providing non-parametric context (decision gate: Z-score).",
            ]
            if driver_resources:
                evidence.append(f"Primary spending drivers: {', '.join(driver_resources)}.")

            dim_label = (
                f"Service '{service_name}'"
                if service_name
                else (
                    f"Account '{account_id}'"
                    if account_id
                    else (f"Region '{region}'" if region else "Overall cloud infrastructure")
                )
            )
            explanation = (
                f"{dim_label} experienced an abnormal spending surge of +${delta:,.2f} {currency} "
                f"(+{pct_increase:.1f}%) on {test_date.isoformat()} exceeding the {len(baseline_slice)}-day baseline ({z_score:.1f}σ)."
            )

            detected.append(
                CostAnomaly(
                    detected_date=test_date,
                    provider=provider_val,
                    account_id=account_id,
                    service_name=service_name,
                    region=region,
                    resource_id=driver_resources[0] if len(driver_resources) == 1 else None,
                    actual_cost=round(test_spend, 2),
                    expected_cost=round(mean, 2),
                    cost_delta=round(delta, 2),
                    percentage_increase=round(pct_increase, 1),
                    z_score=round(z_score, 2),
                    severity=severity,
                    confidence_score=confidence,
                    detection_method="z_score",
                    historical_observation_count=len(baseline_slice),
                    currency=currency,
                    driver_resources=driver_resources,
                    explanation=explanation,
                    evidence=evidence,
                )
            )
