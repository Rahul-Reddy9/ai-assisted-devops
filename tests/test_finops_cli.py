# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""CLI integration tests for `devops-ai finops analyze` command."""

import json
from pathlib import Path
from typer.testing import CliRunner

from devops_ai.cli import app

runner = CliRunner()


def test_cli_finops_help():
    result = runner.invoke(app, ["finops", "--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output


def test_cli_finops_analyze_help():
    result = runner.invoke(app, ["finops", "analyze", "--help"])
    assert result.exit_code == 0
    assert "--billing-file" in result.output
    assert "--group-by" in result.output
    assert "--json" in result.output


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
