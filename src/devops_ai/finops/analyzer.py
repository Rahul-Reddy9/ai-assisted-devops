# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Multi-dimensional cloud expenditure analysis, aggregation, and time-series summaries."""

from collections import defaultdict
import datetime
from typing import Any, Dict, List, Optional

from devops_ai.core.exceptions import CurrencyMismatchError
from devops_ai.core.logging import get_logger
from devops_ai.finops.models import BillingRecord, CostSummary

logger = get_logger("finops.analyzer")


class CostAnalyzer:
    """Performs multi-dimensional aggregation and trend analysis on cloud billing records."""

    @staticmethod
    def filter_records(
        records: List[BillingRecord],
        service: Optional[str] = None,
        account: Optional[str] = None,
        region: Optional[str] = None,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> List[BillingRecord]:
        """Filter billing records by service, account, region, or temporal window."""
        filtered = records

        if service:
            svc_clean = service.strip().lower()
            filtered = [r for r in filtered if svc_clean in r.service_name.lower()]

        if account:
            acct_clean = account.strip().lower()
            filtered = [r for r in filtered if acct_clean in r.account_id.lower()]

        if region:
            reg_clean = region.strip().lower()
            filtered = [r for r in filtered if r.region and reg_clean in r.region.lower()]

        if start_date:
            filtered = [r for r in filtered if r.start_time.date() >= start_date]

        if end_date:
            filtered = [r for r in filtered if r.start_time.date() <= end_date]

        return filtered

    def summarize(
        self,
        records: List[BillingRecord],
        base_currency: Optional[str] = None,
        top_resources_limit: int = 10,
    ) -> CostSummary:
        """Aggregate spending across dimensions and compute financial summaries.

        Args:
            records: List of normalized BillingRecord instances.
            base_currency: Expected currency code (e.g. 'USD'). If omitted, uses the currency of the first record.
            top_resources_limit: Number of highest-spending individual resources to highlight.

        Returns:
            CostSummary instance with total costs and dimensional breakdowns.

        Raises:
            CurrencyMismatchError: If records contain mixed, incompatible currencies.
        """
        if not records:
            return CostSummary(
                total_cost=0.0,
                gross_cost=0.0,
                credit_cost=0.0,
                currency=base_currency or "USD",
                period_start=None,
                period_end=None,
                total_records=0,
                service_costs={},
                account_costs={},
                region_costs={},
                daily_costs={},
                monthly_costs={},
                top_resources=[],
            )

        # 1. Enforce strict currency consistency across all analyzed records
        target_currency = (base_currency or records[0].currency).strip().upper()
        for idx, r in enumerate(records):
            r_curr = r.currency.strip().upper()
            if r_curr != target_currency:
                logger.error(
                    f"Record {idx} ({r.record_id}) currency '{r_curr}' does not match expected '{target_currency}'"
                )
                raise CurrencyMismatchError(base_currency=target_currency, found_currency=r_curr)

        # 2. Accumulate dimensional aggregates
        total_cost = 0.0
        gross_cost = 0.0
        credit_cost = 0.0
        service_costs: Dict[str, float] = defaultdict(float)
        account_costs: Dict[str, float] = defaultdict(float)
        region_costs: Dict[str, float] = defaultdict(float)
        daily_costs: Dict[str, float] = defaultdict(float)
        monthly_costs: Dict[str, float] = defaultdict(float)
        resource_spend: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "service": "unknown", "name": None}
        )

        earliest_start: Optional[datetime.datetime] = None
        latest_end: Optional[datetime.datetime] = None

        for r in records:
            cost = float(r.cost)
            total_cost += cost
            if cost > 0.0:
                gross_cost += cost
            elif cost < 0.0:
                credit_cost += cost

            # Temporal tracking
            if earliest_start is None or r.start_time < earliest_start:
                earliest_start = r.start_time
            if latest_end is None or r.end_time > latest_end:
                latest_end = r.end_time

            # Service, account, region
            service_costs[r.service_name] += cost
            account_costs[r.account_id] += cost
            reg_key = r.region or "global"
            region_costs[reg_key] += cost

            # Time series aggregation
            day_str = r.start_time.strftime("%Y-%m-%d")
            daily_costs[day_str] += cost

            month_str = r.start_time.strftime("%Y-%m")
            monthly_costs[month_str] += cost

            # Resource aggregation
            res_id = r.resource_id or f"unspecified-{r.service_name}"
            entry = resource_spend[res_id]
            entry["cost"] += cost
            entry["service"] = r.service_name
            if r.resource_name:
                entry["name"] = r.resource_name

        # 3. Sort and round top spending resources
        sorted_resources = sorted(
            resource_spend.items(), key=lambda item: item[1]["cost"], reverse=True
        )[:top_resources_limit]

        top_resources_list: List[Dict[str, Any]] = []
        for res_id, data in sorted_resources:
            r_cost = round(data["cost"], 2)
            pct = round((r_cost / total_cost) * 100.0, 1) if total_cost > 0 else 0.0
            top_resources_list.append(
                {
                    "resource_id": res_id,
                    "resource_name": data["name"],
                    "service_name": data["service"],
                    "cost": r_cost,
                    "percentage": pct,
                }
            )

        # 4. Clean and round dictionaries (sorted by cost descending or chronological keys)
        sorted_services = {
            k: round(v, 2)
            for k, v in sorted(service_costs.items(), key=lambda x: x[1], reverse=True)
        }
        sorted_accounts = {
            k: round(v, 2)
            for k, v in sorted(account_costs.items(), key=lambda x: x[1], reverse=True)
        }
        sorted_regions = {
            k: round(v, 2)
            for k, v in sorted(region_costs.items(), key=lambda x: x[1], reverse=True)
        }
        sorted_daily = {k: round(v, 2) for k, v in sorted(daily_costs.items())}
        sorted_monthly = {k: round(v, 2) for k, v in sorted(monthly_costs.items())}

        return CostSummary(
            total_cost=round(total_cost, 2),
            gross_cost=round(gross_cost, 2),
            credit_cost=round(credit_cost, 2),
            currency=target_currency,
            period_start=earliest_start,
            period_end=latest_end,
            total_records=len(records),
            service_costs=sorted_services,
            account_costs=sorted_accounts,
            region_costs=sorted_regions,
            daily_costs=sorted_daily,
            monthly_costs=sorted_monthly,
            top_resources=top_resources_list,
        )
