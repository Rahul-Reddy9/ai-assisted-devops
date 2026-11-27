# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""CLI integration tests for `devops-ai finops analyze` command."""

import json
from pathlib import Path
import re
from typer.testing import CliRunner

from devops_ai.cli import app
from devops_ai.observability.metrics import FINOPS_OPERATIONS_TOTAL

runner = CliRunner()


def _clean_output(text: str) -> str:
    """Strip ANSI escape sequences from CLI output for robust substring assertions."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def test_cli_finops_help():
    result = runner.invoke(app, ["finops", "--help"])
    assert result.exit_code == 0
    assert "analyze" in _clean_output(result.output)


def test_cli_finops_analyze_help():
    result = runner.invoke(app, ["finops", "analyze", "--help"])
    assert result.exit_code == 0
    clean = _clean_output(result.output)
    assert "--billing-file" in clean
    assert "--group-by" in clean
    assert "--json" in clean


def test_cli_finops_analyze_default_synthetic():
    result = runner.invoke(app, ["finops", "analyze"])
    assert result.exit_code == 0
    assert "Cloud FinOps Cost Analysis" in result.output
    assert "SYNTHETIC_BENCHMARK" in result.output
    assert "Cost by Cloud Service" in result.output
    assert "Top Spending Resources" in result.output


def test_cli_finops_analyze_json():
    result = runner.invoke(app, ["finops", "analyze", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "total_cost" in parsed
    assert "currency" in parsed
    assert parsed["currency"] == "USD"
    assert "service_costs" in parsed
    assert parsed["total_records"] > 0


def test_cli_finops_analyze_group_by_dimensions():
    # Test grouping by account
    res_acct = runner.invoke(app, ["finops", "analyze", "--group-by", "account"])
    assert res_acct.exit_code == 0
    assert "Cost by Cloud Account" in res_acct.output

    # Test grouping by region
    res_reg = runner.invoke(app, ["finops", "analyze", "--group-by", "region"])
    assert res_reg.exit_code == 0
    assert "Cost by Cloud Region" in res_reg.output

    # Test grouping by date
    res_date = runner.invoke(app, ["finops", "analyze", "--group-by", "date"])
    assert res_date.exit_code == 0
    assert "Daily Spend Breakdown" in res_date.output


def test_cli_finops_analyze_service_filter():
    result = runner.invoke(app, ["finops", "analyze", "--service", "AmazonEC2", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed["service_costs"]) == 1
    assert "AmazonEC2" in parsed["service_costs"]


def test_cli_finops_analyze_custom_file(tmp_path: Path):
    csv_file = tmp_path / "custom_billing.csv"
    csv_file.write_text("""account_id,service_name,resource_id,cost,currency,start_time,end_time
corp-1,CloudStorage,bucket-x,250.00,USD,2026-11-01,2026-11-02
""")
    result = runner.invoke(app, ["finops", "analyze", "--billing-file", str(csv_file), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["total_cost"] == 250.00
    assert parsed["total_records"] == 1

    # Verify --file alias works identically
    alias_res = runner.invoke(app, ["finops", "analyze", "--file", str(csv_file), "--json"])
    assert alias_res.exit_code == 0
    alias_parsed = json.loads(alias_res.output)
    assert alias_parsed["total_cost"] == 250.00


def test_cli_finops_analyze_date_filtering(tmp_path: Path):
    csv_file = tmp_path / "dated_billing.csv"
    csv_file.write_text("""account_id,service_name,cost,currency,start_time,end_time
1,EC2,10.0,USD,2026-11-01T00:00:00Z,2026-11-02T00:00:00Z
1,EC2,20.0,USD,2026-11-15T00:00:00Z,2026-11-16T00:00:00Z
1,EC2,30.0,USD,2026-11-25T00:00:00Z,2026-11-26T00:00:00Z
""")
    res = runner.invoke(
        app,
        [
            "finops",
            "analyze",
            "--file",
            str(csv_file),
            "--start-date",
            "2026-11-10",
            "--end-date",
            "2026-11-20",
            "--json",
        ],
    )
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert parsed["total_records"] == 1
    assert parsed["total_cost"] == 20.0


def test_cli_finops_analyze_invalid_date_formats():
    res_bad_start = runner.invoke(app, ["finops", "analyze", "--start-date", "invalid-date"])
    assert res_bad_start.exit_code == 1
    assert "Invalid Date Format" in res_bad_start.output

    res_bad_end = runner.invoke(app, ["finops", "analyze", "--end-date", "2026/11/01"])
    assert res_bad_end.exit_code == 1
    assert "Invalid Date Format" in res_bad_end.output


def test_cli_finops_analyze_invalid_date_range():
    res = runner.invoke(
        app,
        ["finops", "analyze", "--start-date", "2026-12-01", "--end-date", "2026-11-01"],
    )
    assert res.exit_code == 1
    assert "Invalid Date Range" in res.output


def test_cli_finops_analyze_with_credits(tmp_path: Path):
    csv_file = tmp_path / "credits_billing.csv"
    csv_file.write_text("""account_id,service_name,cost,currency,start_time,end_time
1,AmazonEC2,150.00,USD,2026-11-01T00:00:00Z,2026-11-02T00:00:00Z
1,AmazonEC2,-50.00,USD,2026-11-01T00:00:00Z,2026-11-02T00:00:00Z
""")
    res = runner.invoke(app, ["finops", "analyze", "--file", str(csv_file)])
    assert res.exit_code == 0
    assert "Net Spend:" in res.output
    assert "$100.00 USD" in res.output
    assert "Gross Charges:" in res.output
    assert "$150.00 USD" in res.output
    assert "Credits & Refunds:" in res.output
    assert "-$50.00 USD" in res.output


def test_cli_finops_analyze_output_file(tmp_path: Path):
    out_file = tmp_path / "cost_report.json"
    result = runner.invoke(app, ["finops", "analyze", "--output", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert "total_cost" in parsed


def test_cli_finops_analyze_missing_file(tmp_path: Path):
    result = runner.invoke(
        app, ["finops", "analyze", "--billing-file", str(tmp_path / "missing.csv")]
    )
    assert result.exit_code == 1
    assert "Billing Data Error" in result.output


def test_cli_finops_optimize_help():
    res = runner.invoke(app, ["finops", "optimize", "--help"])
    assert res.exit_code == 0
    clean = _clean_output(res.output)
    assert "--inventory-file" in clean
    assert "--utilization-file" in clean
    assert "--min-savings" in clean
    assert "--risk" in clean


def test_cli_finops_optimize_default_synthetic():
    res = runner.invoke(app, ["finops", "optimize"])
    assert res.exit_code == 0
    assert "Cloud FinOps Rightsizing & Optimization Report" in res.output
    assert "Total Potential Monthly Savings:" in res.output
    assert "Guaranteed Savings" in res.output
    assert "Estimated Savings" in res.output
    assert "Actionable Optimization Recommendations" in res.output
    assert "Advisory Notice:" in res.output


def test_cli_finops_optimize_json():
    res = runner.invoke(app, ["finops", "optimize", "--json"])
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert "total_monthly_savings" in parsed
    assert "total_annual_savings" in parsed
    assert "recommendations_count" in parsed
    assert parsed["recommendations_count"] > 0
    assert len(parsed["recommendations"]) == parsed["recommendations_count"]


def test_cli_finops_optimize_filters():
    # Filter by category storage
    res_storage = runner.invoke(app, ["finops", "optimize", "--category", "storage", "--json"])
    assert res_storage.exit_code == 0
    parsed_storage = json.loads(res_storage.output)
    assert parsed_storage["recommendations_count"] == 1
    assert parsed_storage["recommendations"][0]["action"] == "CLEANUP_ORPHAN"

    # Filter by min savings ($100/mo)
    res_savings = runner.invoke(app, ["finops", "optimize", "--min-savings", "100.0", "--json"])
    assert res_savings.exit_code == 0
    parsed_savings = json.loads(res_savings.output)
    for r in parsed_savings["recommendations"]:
        assert r["estimated_monthly_savings"] >= 100.0


def test_cli_finops_optimize_custom_files(tmp_path: Path):
    inv_file = tmp_path / "inventory.json"
    inv_file.write_text("""[
  {
    "resource_id": "custom-idle-node",
    "provider": "aws",
    "account_id": "111",
    "service_name": "AmazonEC2",
    "category": "compute",
    "monthly_cost": 200.0,
    "currency": "USD"
  }
]""")
    util_file = tmp_path / "telemetry.json"
    util_file.write_text("""[
  {
    "resource_id": "custom-idle-node",
    "service_name": "AmazonEC2",
    "timestamp": "2026-11-20T00:00:00Z",
    "avg_cpu_pct": 0.5,
    "peak_cpu_pct": 1.2,
    "network_in_bytes": 0.0,
    "network_out_bytes": 0.0,
    "is_attached": true
  }
]""")
    res = runner.invoke(
        app,
        [
            "finops",
            "optimize",
            "--inventory-file",
            str(inv_file),
            "--utilization-file",
            str(util_file),
            "--json",
        ],
    )
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert parsed["recommendations_count"] == 1
    assert parsed["recommendations"][0]["action"] == "TERMINATE"
    assert parsed["recommendations"][0]["estimated_monthly_savings"] == 200.0


def test_cli_finops_optimize_output_file(tmp_path: Path):
    out_file = tmp_path / "opt_report.json"
    res = runner.invoke(app, ["finops", "optimize", "--output", str(out_file)])
    assert res.exit_code == 0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert "total_monthly_savings" in parsed


def test_cli_finops_optimize_invalid_arguments():
    res_bad_prov = runner.invoke(app, ["finops", "optimize", "--provider", "invalid_cloud"])
    assert res_bad_prov.exit_code == 1
    assert "Unknown provider" in res_bad_prov.output

    res_bad_cat = runner.invoke(app, ["finops", "optimize", "--category", "invalid_category"])
    assert res_bad_cat.exit_code == 1
    assert "Unknown category" in res_bad_cat.output


# ==============================================================================
# FinOps Anomalies CLI Tests
# ==============================================================================


def test_cli_finops_anomalies_help():
    result = runner.invoke(app, ["finops", "anomalies", "--help"])
    assert result.exit_code == 0
    clean = _clean_output(result.output)
    assert "--billing-file" in clean
    assert "--service" in clean
    assert "--severity" in clean
    assert "--z-threshold" in clean
    assert "--json" in clean


def test_cli_finops_anomalies_default_synthetic():
    result = runner.invoke(app, ["finops", "anomalies"])
    assert result.exit_code == 0
    assert "Cost Anomaly Detection Report" in result.output
    assert "Detected Statistical Spending Surges" in result.output
    assert "Advisory Notice" in result.output


def test_cli_finops_anomalies_json():
    result = runner.invoke(app, ["finops", "anomalies", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "total_anomalies" in parsed
    assert "total_excess_spend" in parsed
    assert "anomalies" in parsed
    assert parsed["total_anomalies"] > 0


def test_cli_finops_anomalies_filters():
    # Service filter
    res_svc = runner.invoke(app, ["finops", "anomalies", "--service", "AmazonEC2", "--json"])
    assert res_svc.exit_code == 0
    parsed_svc = json.loads(res_svc.output)
    for a in parsed_svc["anomalies"]:
        if a["service_name"]:
            assert a["service_name"] == "AmazonEC2"

    # Severity filter
    res_sev = runner.invoke(app, ["finops", "anomalies", "--severity", "CRITICAL", "--json"])
    assert res_sev.exit_code == 0
    parsed_sev = json.loads(res_sev.output)
    for a in parsed_sev["anomalies"]:
        assert a["severity"] == "CRITICAL"


def test_cli_finops_anomalies_output_file(tmp_path: Path):
    out_file = tmp_path / "anom_report.json"
    res = runner.invoke(app, ["finops", "anomalies", "--output", str(out_file)])
    assert res.exit_code == 0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert "total_anomalies" in parsed


def test_cli_finops_anomalies_invalid_inputs():
    res_bad_date = runner.invoke(app, ["finops", "anomalies", "--start-date", "invalid-date"])
    assert res_bad_date.exit_code == 1
    assert "Invalid --start-date format" in res_bad_date.output

    res_range = runner.invoke(
        app, ["finops", "anomalies", "--start-date", "2026-11-20", "--end-date", "2026-11-10"]
    )
    assert res_range.exit_code == 1
    assert "cannot be later than" in res_range.output

    res_bad_sev = runner.invoke(app, ["finops", "anomalies", "--severity", "EXTREME"])
    assert res_bad_sev.exit_code == 1
    assert "Invalid --severity" in res_bad_sev.output


# ==============================================================================
# FinOps Forecast CLI Tests
# ==============================================================================


def test_cli_finops_forecast_help():
    result = runner.invoke(app, ["finops", "forecast", "--help"])
    assert result.exit_code == 0
    clean = _clean_output(result.output)
    assert "--horizon" in clean
    assert "--frequency" in clean
    assert "--method" in clean
    assert "--confidence" in clean
    assert "--json" in clean


def test_cli_finops_forecast_default_synthetic():
    result = runner.invoke(app, ["finops", "forecast"])
    assert result.exit_code == 0
    assert "Expenditure Forecast Report" in result.output
    assert "Projected Spend Trajectory" in result.output
    assert "In-Sample Goodness of Fit" in result.output
    assert "Advisory Notice" in result.output


def test_cli_finops_forecast_json():
    result = runner.invoke(app, ["finops", "forecast", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "forecast_id" in parsed
    assert "total_projected_cost" in parsed
    assert "total_p10_cost" in parsed
    assert "total_p90_cost" in parsed
    assert (
        0.0
        <= parsed["total_p10_cost"]
        <= parsed["total_projected_cost"]
        <= parsed["total_p90_cost"]
    )


def test_cli_finops_forecast_methods_and_frequency():
    # Test methods
    methods = [
        "linear_regression",
        "weighted_moving_average",
        "exponential_smoothing",
        "historical_average",
    ]
    for m in methods:
        res = runner.invoke(app, ["finops", "forecast", "--method", m, "--horizon", "7", "--json"])
        assert res.exit_code == 0
        parsed = json.loads(res.output)
        assert parsed["horizon_days"] == 7
        assert parsed["method"] == m

    # Monthly frequency
    res_monthly = runner.invoke(
        app, ["finops", "forecast", "--frequency", "monthly", "--horizon", "60", "--json"]
    )
    assert res_monthly.exit_code == 0
    parsed_monthly = json.loads(res_monthly.output)
    assert parsed_monthly["horizon_days"] >= 2

    # Monthly frequency text output
    res_monthly_text = runner.invoke(
        app, ["finops", "forecast", "--frequency", "monthly", "--horizon", "60"]
    )
    assert res_monthly_text.exit_code == 0
    assert "Projected Monthly Average:" in res_monthly_text.output


def test_cli_finops_forecast_output_file(tmp_path: Path):
    out_file = tmp_path / "fcst_report.json"
    res = runner.invoke(app, ["finops", "forecast", "--output", str(out_file)])
    assert res.exit_code == 0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert "total_projected_cost" in parsed


def test_cli_finops_forecast_invalid_inputs():
    res_freq = runner.invoke(app, ["finops", "forecast", "--frequency", "yearly"])
    assert res_freq.exit_code == 1
    assert "Invalid --frequency" in res_freq.output

    res_method = runner.invoke(app, ["finops", "forecast", "--method", "arima"])
    assert res_method.exit_code == 1
    assert "Invalid --method" in res_method.output

    res_horizon = runner.invoke(app, ["finops", "forecast", "--horizon", "0"])
    assert res_horizon.exit_code == 1
    assert "Invalid --horizon" in res_horizon.output

    res_conf = runner.invoke(app, ["finops", "forecast", "--confidence", "0.20"])
    assert res_conf.exit_code == 1
    assert "Invalid --confidence" in res_conf.output


def test_cli_finops_advise_help():
    res = runner.invoke(app, ["finops", "advise", "--help"])
    assert res.exit_code == 0
    clean = _clean_output(res.output)
    assert "--billing-file" in clean
    assert "--inventory-file" in clean
    assert "--utilization-file" in clean
    assert "--offline" in clean
    assert "--query" in clean
    assert "--output" in clean
    assert "--json" in clean


def test_cli_finops_advise_offline_default():
    res = runner.invoke(app, ["finops", "advise", "--offline"])
    assert res.exit_code == 0
    assert "AI-Assisted FinOps Strategic Advisory Report" in res.output
    assert "Total Identified Monthly Savings" in res.output
    assert "Advisory Disclaimer" in res.output
    assert "Prioritized Cost Optimization Recommendations" in res.output


def test_cli_finops_advise_offline_json():
    res = runner.invoke(app, ["finops", "advise", "--offline", "--json"])
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert "executive_summary" in parsed
    assert "total_monthly_savings" in parsed
    assert "guaranteed_monthly_savings" in parsed
    assert "estimated_monthly_savings" in parsed
    assert "recommendations" in parsed
    assert len(parsed["recommendations"]) > 0
    assert parsed["provider"] == "offline"


def test_cli_finops_advise_output_file(tmp_path: Path):
    out_file = tmp_path / "advisory_report.json"
    res = runner.invoke(app, ["finops", "advise", "--offline", "--output", str(out_file)])
    assert res.exit_code == 0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert "total_monthly_savings" in parsed
    assert len(parsed["recommendations"]) > 0


def test_cli_finops_advise_custom_files(tmp_path: Path):
    # Minimal custom billing file
    billing_data = [
        {
            "record_id": "rec-1",
            "provider": "aws",
            "account_id": "acc-1",
            "service_name": "AmazonEC2",
            "resource_id": "i-01",
            "region": "us-east-1",
            "cost_amount": 100.0,
            "currency": "USD",
            "start_time": "2026-11-01T00:00:00Z",
            "end_time": "2026-11-01T23:59:59Z",
        }
    ]
    b_file = tmp_path / "billing.json"
    b_file.write_text(json.dumps(billing_data), encoding="utf-8")

    # Minimal custom inventory file
    inv_data = [
        {
            "resource_id": "vol-orphan",
            "resource_name": "unattached-vol",
            "provider": "aws",
            "category": "storage",
            "region": "us-east-1",
            "service_name": "AmazonEC2",
            "monthly_cost": 50.0,
            "currency": "USD",
            "is_attached": False,
        }
    ]
    i_file = tmp_path / "inventory.json"
    i_file.write_text(json.dumps(inv_data), encoding="utf-8")

    # Telemetry demonstrating unattached volume
    util_data = [
        {
            "resource_id": "vol-orphan",
            "service_name": "AmazonEC2",
            "timestamp": "2026-11-20T00:00:00Z",
            "is_attached": False,
            "disk_read_bytes_per_sec": 0.0,
            "disk_write_bytes_per_sec": 0.0,
        }
    ]
    u_file = tmp_path / "utilization.json"
    u_file.write_text(json.dumps(util_data), encoding="utf-8")

    res = runner.invoke(
        app,
        [
            "finops",
            "advise",
            "--offline",
            "--billing-file",
            str(b_file),
            "--inventory-file",
            str(i_file),
            "--utilization-file",
            str(u_file),
            "--query",
            "Focus on orphan disk cleanup",
            "--json",
        ],
    )
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert parsed["guaranteed_monthly_savings"] == 50.0
    assert any("vol-orphan" in r.get("affected_resources", []) for r in parsed["recommendations"])


def test_cli_finops_telemetry_parity_analyze():
    """Verify operation telemetry is recorded accurately on success and failure for analyze (F-05)."""
    succ_before = FINOPS_OPERATIONS_TOTAL.labels(operation="analyze", status="success")._value.get()
    fail_before = FINOPS_OPERATIONS_TOTAL.labels(operation="analyze", status="failure")._value.get()

    res_ok = runner.invoke(app, ["finops", "analyze", "--json"])
    assert res_ok.exit_code == 0
    succ_after = FINOPS_OPERATIONS_TOTAL.labels(operation="analyze", status="success")._value.get()
    assert succ_after == succ_before + 1

    res_err = runner.invoke(app, ["finops", "analyze", "--start-date", "invalid-date-format"])
    assert res_err.exit_code != 0
    fail_after = FINOPS_OPERATIONS_TOTAL.labels(operation="analyze", status="failure")._value.get()
    assert fail_after == fail_before + 1


def test_cli_finops_telemetry_parity_optimize():
    """Verify operation telemetry is recorded accurately on success and failure for optimize (F-05)."""
    succ_before = FINOPS_OPERATIONS_TOTAL.labels(
        operation="optimize", status="success"
    )._value.get()
    fail_before = FINOPS_OPERATIONS_TOTAL.labels(
        operation="optimize", status="failure"
    )._value.get()

    res_ok = runner.invoke(app, ["finops", "optimize", "--json"])
    assert res_ok.exit_code == 0
    succ_after = FINOPS_OPERATIONS_TOTAL.labels(operation="optimize", status="success")._value.get()
    assert succ_after == succ_before + 1

    res_err = runner.invoke(app, ["finops", "optimize", "--provider", "unsupported_cloud_xyz"])
    assert res_err.exit_code != 0
    fail_after = FINOPS_OPERATIONS_TOTAL.labels(operation="optimize", status="failure")._value.get()
    assert fail_after == fail_before + 1


def test_cli_finops_telemetry_parity_anomalies():
    """Verify operation telemetry is recorded accurately on success and failure for anomalies (F-05)."""
    succ_before = FINOPS_OPERATIONS_TOTAL.labels(
        operation="anomalies", status="success"
    )._value.get()
    fail_before = FINOPS_OPERATIONS_TOTAL.labels(
        operation="anomalies", status="failure"
    )._value.get()

    res_ok = runner.invoke(app, ["finops", "anomalies", "--json"])
    assert res_ok.exit_code == 0
    succ_after = FINOPS_OPERATIONS_TOTAL.labels(
        operation="anomalies", status="success"
    )._value.get()
    assert succ_after == succ_before + 1

    res_err = runner.invoke(app, ["finops", "anomalies", "--severity", "invalid_severity_xyz"])
    assert res_err.exit_code != 0
    fail_after = FINOPS_OPERATIONS_TOTAL.labels(
        operation="anomalies", status="failure"
    )._value.get()
    assert fail_after == fail_before + 1


def test_cli_finops_telemetry_parity_forecast():
    """Verify operation telemetry is recorded accurately on success and failure for forecast (F-05)."""
    succ_before = FINOPS_OPERATIONS_TOTAL.labels(
        operation="forecast", status="success"
    )._value.get()
    fail_before = FINOPS_OPERATIONS_TOTAL.labels(
        operation="forecast", status="failure"
    )._value.get()

    res_ok = runner.invoke(app, ["finops", "forecast", "--json"])
    assert res_ok.exit_code == 0
    succ_after = FINOPS_OPERATIONS_TOTAL.labels(operation="forecast", status="success")._value.get()
    assert succ_after == succ_before + 1

    res_err = runner.invoke(app, ["finops", "forecast", "--frequency", "hourly"])
    assert res_err.exit_code != 0
    fail_after = FINOPS_OPERATIONS_TOTAL.labels(operation="forecast", status="failure")._value.get()
    assert fail_after == fail_before + 1


def test_cli_finops_repeated_invocation_telemetry():
    """Verify repeated invocations increment operation counter without double-counting (F-05)."""
    before = FINOPS_OPERATIONS_TOTAL.labels(operation="analyze", status="success")._value.get()
    for _ in range(3):
        res = runner.invoke(app, ["finops", "analyze", "--json"])
        assert res.exit_code == 0
    after = FINOPS_OPERATIONS_TOTAL.labels(operation="analyze", status="success")._value.get()
    assert after == before + 3
