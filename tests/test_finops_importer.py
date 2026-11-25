# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit and integration tests for secure FinOps billing data importers and multi-schema normalizers."""

from pathlib import Path
import pytest

from devops_ai.core.exceptions import BillingDataError
from devops_ai.finops.importer import (
    get_synthetic_billing_dataset,
    load_billing_records,
    parse_billing_records_from_csv,
    parse_billing_records_from_json,
)
from devops_ai.finops.models import CloudProvider


class TestBillingImporters:
    def test_standard_csv_import(self):
        csv_data = """account_id,service_name,resource_id,cost,currency,start_time,end_time,provider
acct-101,AmazonEC2,i-01234,42.50,USD,2026-11-01T00:00:00Z,2026-11-02T00:00:00Z,aws
acct-101,AmazonS3,bucket-01,15.20,USD,2026-11-01T00:00:00Z,2026-11-02T00:00:00Z,aws
"""
        records, schema = parse_billing_records_from_csv(csv_data)
        assert schema == "STANDARD"
        assert len(records) == 2
        assert records[0].cost == 42.50
        assert records[0].service_name == "AmazonEC2"
        assert records[0].provider == CloudProvider.AWS

    def test_standard_json_import(self):
        json_data = """[
  {
    "account_id": "proj-99",
    "service_name": "Compute Engine",
    "resource_id": "instance-1",
    "cost": 88.00,
    "currency": "USD",
    "start_time": "2026-11-01T00:00:00Z",
    "end_time": "2026-11-02T00:00:00Z",
    "provider": "gcp"
  }
]"""
        records, schema = parse_billing_records_from_json(json_data)
        assert len(records) == 1
        assert records[0].cost == 88.00
        assert records[0].provider == CloudProvider.GCP

    def test_aws_cur_schema_normalization(self):
        aws_cur_csv = """lineItem/UsageAccountId,lineItem/ProductCode,lineItem/ResourceId,lineItem/UnblendedCost,lineItem/CurrencyCode,lineItem/UsageStartDate,lineItem/UsageEndDate,product/region
112233445566,AmazonEC2,i-099887766,125.75,USD,2026-11-01 00:00:00,2026-11-02 00:00:00,us-east-1
"""
        records, schema = parse_billing_records_from_csv(aws_cur_csv)
        assert schema == "AWS_CUR"
        assert len(records) == 1
        rec = records[0]
        assert rec.provider == CloudProvider.AWS
        assert rec.account_id == "112233445566"
        assert rec.service_name == "AmazonEC2"
        assert rec.resource_id == "i-099887766"
        assert rec.cost == 125.75
        assert rec.region == "us-east-1"

    def test_azure_export_schema_normalization(self):
        azure_csv = """SubscriptionId,ConsumedService,ResourceId,CostInBillingCurrency,BillingCurrency,Date,ResourceLocation
sub-az-111,Virtual Machines,/subscriptions/sub-az-111/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachines/vm1,95.40,USD,2026-11-05,eastus
"""
        records, schema = parse_billing_records_from_csv(azure_csv)
        assert schema == "AZURE_EXPORT"
        assert len(records) == 1
        rec = records[0]
        assert rec.provider == CloudProvider.AZURE
        assert rec.account_id == "sub-az-111"
        assert rec.service_name == "Virtual Machines"
        assert rec.cost == 95.40
        assert rec.region == "eastus"

    def test_gcp_billing_schema_normalization(self):
        gcp_csv = """project.id,service.description,resource.name,cost,currency,usage_start_time,location.region
gcp-prod-infra,Cloud Storage,prod-bucket-artifacts,64.20,USD,2026-11-01T00:00:00Z,us-central1
"""
        records, schema = parse_billing_records_from_csv(gcp_csv)
        assert schema == "GCP_EXPORT"
        assert len(records) == 1
        rec = records[0]
        assert rec.provider == CloudProvider.GCP
        assert rec.account_id == "gcp-prod-infra"
        assert rec.service_name == "Cloud Storage"
        assert rec.cost == 64.20

        gcp_json = """[
  {
    "project.id": "gcp-prod-infra",
    "service.description": "Cloud Storage",
    "resource.name": "prod-bucket-artifacts",
    "cost": 64.20,
    "currency": "USD",
    "usage_start_time": "2026-11-01T00:00:00Z",
    "location.region": "us-central1"
  }
]"""
        json_recs, json_schema = parse_billing_records_from_json(gcp_json)
        assert json_schema == "GCP_EXPORT"
        assert len(json_recs) == 1

    def test_path_traversal_rejection(self, tmp_path: Path):
        allowed_dir = tmp_path / "allowed_billing"
        allowed_dir.mkdir()
        outside_file = tmp_path / "outside.csv"
        outside_file.write_text("account_id,service_name,cost,start_time\n1,s,10,2026-01-01\n")

        with pytest.raises(BillingDataError, match="breaches permitted root"):
            load_billing_records(outside_file, allowed_root=allowed_dir)

    def test_nonexistent_file(self, tmp_path: Path):
        with pytest.raises(BillingDataError, match="does not exist"):
            load_billing_records(tmp_path / "nonexistent.csv")

    def test_empty_file_rejected(self, tmp_path: Path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("   \n")
        with pytest.raises(BillingDataError, match="empty"):
            load_billing_records(empty_csv)

    def test_configurable_file_size_limit(self, tmp_path: Path):
        test_file = tmp_path / "sized_test.csv"
        # Create a file roughly 2KB
        header = "account_id,service_name,cost,start_time\n"
        row = "acct-1,EC2,10.0,2026-11-01\n"
        test_file.write_text(header + row * 60)
        file_size = test_file.stat().st_size
        assert file_size > 1024

        # Custom smaller limit (1024 bytes) triggers error
        with pytest.raises(BillingDataError, match="exceeds maximum limit"):
            load_billing_records(test_file, max_file_size_bytes=1024)

        # Custom larger limit permits load
        recs, schema = load_billing_records(test_file, max_file_size_bytes=file_size + 1000)
        assert len(recs) == 60

    def test_import_negative_costs_and_credits(self):
        csv_with_credits = """account_id,service_name,resource_id,cost,currency,start_time,end_time
acct-1,AmazonEC2,i-12345,100.00,USD,2026-11-01,2026-11-02
acct-1,AmazonEC2,i-12345,-25.00,USD,2026-11-01,2026-11-02
"""
        records, schema = parse_billing_records_from_csv(csv_with_credits)
        assert len(records) == 2
        assert records[0].cost == 100.00
        assert records[0].is_credit is False
        assert records[1].cost == -25.00
        assert records[1].is_credit is True

    def test_synthetic_benchmark_dataset(self):
        records = get_synthetic_billing_dataset()
        assert len(records) >= 5
        providers = {r.provider for r in records}
        assert CloudProvider.AWS in providers
        assert CloudProvider.AZURE in providers
        assert CloudProvider.GCP in providers
        for r in records:
            assert r.cost > 0.0
            assert r.currency == "USD"
