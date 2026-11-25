# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Secure cloud billing data importers with multi-schema normalization and path security."""

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from devops_ai.core.exceptions import BillingDataError
from devops_ai.core.logging import get_logger, scrub_secrets
from devops_ai.finops.models import BillingRecord, CloudProvider

logger = get_logger("finops.importer")

MAX_BILLING_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB strict limit


def _parse_iso_or_common_date(val: Any) -> datetime.datetime:
    """Parse various datetime string formats into UTC datetime."""
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=datetime.timezone.utc)
        return val.astimezone(datetime.timezone.utc)

    s = str(val).strip()
    if not s:
        return datetime.datetime.now(datetime.timezone.utc)

    # Standard ISO formats
    iso_candidates = [
        s,
        s.replace("Z", "+00:00"),
    ]
    for cand in iso_candidates:
        try:
            dt = datetime.datetime.fromisoformat(cand)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc)
        except ValueError:
            pass

    # Common date formats in billing exports
    common_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ]
    for fmt in common_formats:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc)
        except ValueError:
            pass

    raise ValueError(f"Unrecognized date/time format: '{s}'")


def _detect_schema(headers_or_keys: List[str]) -> str:
    """Detect the cloud billing schema based on present field keys."""
    norm_keys = {k.strip().lower() for k in headers_or_keys}

    # AWS Cost and Usage Report (CUR) indicators
    aws_indicators = {"lineitem/unblendedcost", "lineitem/productcode", "lineitem/usageaccountid"}
    if any(k in norm_keys for k in aws_indicators):
        return "AWS_CUR"

    # Azure Cost Export indicators
    azure_indicators = {
        "consumedservice",
        "costinbillingcurrency",
        "subscriptionguid",
        "subscriptionid",
    }
    if any(k in norm_keys for k in azure_indicators):
        return "AZURE_EXPORT"

    # GCP Cloud Billing Export indicators
    gcp_indicators = {
        "service/description",
        "service.description",
        "project/id",
        "project.id",
        "usage_start_time",
    }
    if any(k in norm_keys for k in gcp_indicators):
        return "GCP_EXPORT"

    return "STANDARD"


def _normalize_record_dict(row: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
    """Map provider-specific export line item dictionary into standard BillingRecord schema."""
    norm_map = {k.strip().lower(): v for k, v in row.items()}

    def get_first(*keys: str, default: Any = None) -> Any:
        for k in keys:
            k_low = k.lower()
            if k_low in norm_map and norm_map[k_low] not in (None, ""):
                return norm_map[k_low]
        return default

    cost_val = 0.0
    raw_cost = get_first(
        "cost",
        "lineitem/unblendedcost",
        "lineitem/blendedcost",
        "costinbillingcurrency",
        "pretaxcost",
        default=0.0,
    )
    try:
        cost_val = float(raw_cost)
    except (ValueError, TypeError):
        cost_val = 0.0

    currency = (
        str(
            get_first(
                "currency",
                "lineitem/currencycode",
                "billingcurrency",
                default="USD",
            )
        )
        .strip()
        .upper()
    )
    if len(currency) != 3:
        currency = "USD"

    # Time boundaries
    start_time_raw = get_first(
        "start_time",
        "lineitem/usagestartdate",
        "usagestarttime",
        "usage_start_time",
        "date",
        "usagedatetime",
        default=datetime.datetime.now(datetime.timezone.utc),
    )
    start_dt = _parse_iso_or_common_date(start_time_raw)

    end_time_raw = get_first(
        "end_time",
        "lineitem/usageenddate",
        "usageendtime",
        "usage_end_time",
        default=None,
    )
    if end_time_raw:
        end_dt = _parse_iso_or_common_date(end_time_raw)
    else:
        end_dt = start_dt + datetime.timedelta(days=1)

    # Provider and Account
    if schema_type == "AWS_CUR":
        provider = CloudProvider.AWS
        account_id = str(get_first("lineitem/usageaccountid", "account_id", default="aws-unknown"))
        service_name = str(
            get_first(
                "lineitem/productcode", "product/productname", "service_name", default="AWS Service"
            )
        )
        resource_id = get_first("lineitem/resourceid", "resource_id", default=None)
        region = get_first("product/region", "lineitem/availabilityzone", "region", default=None)
        usage_quantity = float(
            get_first("lineitem/usageamount", "usage_quantity", default=0.0) or 0.0
        )
        usage_unit = get_first("pricing/unit", "usage_unit", default=None)
    elif schema_type == "AZURE_EXPORT":
        provider = CloudProvider.AZURE
        account_id = str(
            get_first("subscriptionid", "subscriptionguid", "account_id", default="azure-unknown")
        )
        service_name = str(
            get_first("consumedservice", "servicename", "service_name", default="Azure Service")
        )
        resource_id = get_first("resourceid", "resource_id", default=None)
        region = get_first("resourcelocation", "region", default=None)
        usage_quantity = float(get_first("quantity", "usage_quantity", default=0.0) or 0.0)
        usage_unit = get_first("unitofmeasure", "usage_unit", default=None)
    elif schema_type == "GCP_EXPORT":
        provider = CloudProvider.GCP
        account_id = str(
            get_first("project/id", "project.id", "project_id", "account_id", default="gcp-unknown")
        )
        service_name = str(
            get_first(
                "service/description", "service.description", "service_name", default="GCP Service"
            )
        )
        resource_id = get_first("resource/name", "resource.name", "resource_id", default=None)
        region = get_first("location/region", "location.region", "region", default=None)
        usage_quantity = float(get_first("usage/amount", "usage_quantity", default=0.0) or 0.0)
        usage_unit = get_first("usage/unit", "usage_unit", default=None)
    else:
        # Standard schema
        prov_raw = str(get_first("provider", default="custom")).lower()
        provider = (
            CloudProvider(prov_raw)
            if prov_raw in [p.value for p in CloudProvider]
            else CloudProvider.CUSTOM
        )
        account_id = str(get_first("account_id", default="default-account"))
        service_name = str(get_first("service_name", default="General Service"))
        resource_id = get_first("resource_id", default=None)
        region = get_first("region", default=None)
        usage_quantity = float(get_first("usage_quantity", default=0.0) or 0.0)
        usage_unit = get_first("usage_unit", default=None)

    resource_name = get_first("resource_name", default=None)
    usage_type = get_first("usage_type", "lineitem/usagetype", default=None)
    record_id = get_first("record_id", default=None)

    # Tags extraction
    tags: Dict[str, str] = {}
    raw_tags = get_first("tags", default=None)
    if isinstance(raw_tags, dict):
        tags = {str(k): str(v) for k, v in raw_tags.items()}
    elif isinstance(raw_tags, str) and raw_tags.strip().startswith("{"):
        try:
            parsed_tags = json.loads(raw_tags)
            if isinstance(parsed_tags, dict):
                tags = {str(k): str(v) for k, v in parsed_tags.items()}
        except Exception:
            pass

    out: Dict[str, Any] = {
        "provider": provider,
        "account_id": account_id,
        "service_name": service_name,
        "resource_id": str(resource_id) if resource_id else None,
        "resource_name": str(resource_name) if resource_name else None,
        "region": str(region) if region else None,
        "usage_type": str(usage_type) if usage_type else None,
        "usage_quantity": max(0.0, usage_quantity),
        "usage_unit": str(usage_unit) if usage_unit else None,
        "cost": cost_val,
        "currency": currency,
        "start_time": start_dt,
        "end_time": end_dt,
        "tags": tags,
    }
    if record_id:
        out["record_id"] = str(record_id)
    return out


def parse_billing_records_from_csv(
    content: str, max_records: int = 100000
) -> Tuple[List[BillingRecord], str]:
    """Parse CSV content into normalized BillingRecord instances."""
    if not content or not content.strip():
        raise BillingDataError("Billing CSV content is empty", source="csv_reader")

    reader = csv.DictReader(content.splitlines())
    if not reader.fieldnames:
        raise BillingDataError("CSV contains no header row", source="csv_reader")

    detected_schema = _detect_schema(list(reader.fieldnames))
    records: List[BillingRecord] = []

    for row_idx, row in enumerate(reader, start=1):
        if len(records) >= max_records:
            logger.warning(f"Truncated import at maximum allowed records limit ({max_records})")
            break

        # Skip entirely empty rows
        if not any(row.values()):
            continue

        try:
            norm_dict = _normalize_record_dict(row, detected_schema)
            record = BillingRecord.model_validate(norm_dict)
            records.append(record)
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise BillingDataError(
                f"Row {row_idx} failed validation: {sanitized}", source="csv_reader"
            ) from e

    if not records:
        raise BillingDataError("No valid billing records found in CSV", source="csv_reader")

    return records, detected_schema


def parse_billing_records_from_json(
    content: str, max_records: int = 100000
) -> Tuple[List[BillingRecord], str]:
    """Parse JSON content into normalized BillingRecord instances."""
    if not content or not content.strip():
        raise BillingDataError("Billing JSON content is empty", source="json_reader")

    try:
        data = json.loads(content)
    except Exception as e:
        sanitized = scrub_secrets(str(e))
        raise BillingDataError(f"Malformed JSON: {sanitized}", source="json_reader") from e

    # Accept both list of records or dictionary containing a 'records' / 'billing' key
    raw_list: List[Any] = []
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        for k in ("records", "billing", "items", "data"):
            if k in data and isinstance(data[k], list):
                raw_list = data[k]
                break
        if not raw_list:
            raw_list = [data]
    else:
        raise BillingDataError("Root JSON must be a list or object", source="json_reader")

    if not raw_list:
        raise BillingDataError("No billing records found in JSON", source="json_reader")

    # Sample first record keys for schema detection
    first_keys: List[str] = list(raw_list[0].keys()) if isinstance(raw_list[0], dict) else []
    detected_schema = _detect_schema(first_keys)

    records: List[BillingRecord] = []
    for idx, item in enumerate(raw_list, start=1):
        if len(records) >= max_records:
            logger.warning(f"Truncated import at maximum allowed records limit ({max_records})")
            break

        if not isinstance(item, dict):
            raise BillingDataError(
                f"Item {idx} in JSON list is not an object", source="json_reader"
            )

        try:
            norm_dict = _normalize_record_dict(item, detected_schema)
            record = BillingRecord.model_validate(norm_dict)
            records.append(record)
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise BillingDataError(
                f"Item {idx} failed validation: {sanitized}", source="json_reader"
            ) from e

    return records, detected_schema


def load_billing_records(
    path: Union[str, Path],
    allowed_root: Optional[Path] = None,
    max_records: int = 100000,
    max_file_size_bytes: Optional[int] = None,
) -> Tuple[List[BillingRecord], str]:
    """Safely load and validate billing records from a CSV or JSON file.

    Args:
        path: Filesystem path to the billing export file.
        allowed_root: Optional directory boundary to prevent path traversal attacks.
        max_records: Maximum number of records permitted to load.
        max_file_size_bytes: Optional maximum allowed file size in bytes (defaults to 10MB limit).

    Returns:
        Tuple of (List of normalized BillingRecord instances, detected schema string).

    Raises:
        BillingDataError: If file is missing, breaches path bounds, exceeds size limits, or contains invalid data.
    """
    file_path = Path(path)

    # 1. Path existence and boundary checks
    try:
        resolved_path = file_path.resolve()
    except Exception as e:
        raise BillingDataError(f"Invalid filesystem path: {str(e)}", source=str(path)) from e

    if not resolved_path.exists():
        raise BillingDataError(f"Billing file does not exist: '{resolved_path}'", source=str(path))

    if not resolved_path.is_file():
        raise BillingDataError(f"Path is not a regular file: '{resolved_path}'", source=str(path))

    if allowed_root is not None:
        try:
            resolved_root = allowed_root.resolve()
            if not resolved_path.is_relative_to(resolved_root):
                raise BillingDataError(
                    f"Path '{resolved_path}' breaches permitted root directory '{resolved_root}'",
                    source=str(path),
                )
        except AttributeError:
            if resolved_root not in resolved_path.parents and resolved_root != resolved_path:
                raise BillingDataError(
                    f"Path '{resolved_path}' breaches permitted root directory '{resolved_root}'",
                    source=str(path),
                )

    # 2. File size bound check
    file_size = resolved_path.stat().st_size
    effective_max_size = (
        max_file_size_bytes if max_file_size_bytes is not None else MAX_BILLING_FILE_SIZE_BYTES
    )
    if file_size > effective_max_size:
        raise BillingDataError(
            f"File size ({file_size} bytes) exceeds maximum limit ({effective_max_size} bytes)",
            source=str(path),
        )

    # 3. Read content
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except Exception as e:
        raise BillingDataError(f"Failed to read file: {str(e)}", source=str(path)) from e

    if not content.strip():
        raise BillingDataError("Billing file is empty", source=str(path))

    # 4. Parse according to suffix
    suffix = resolved_path.suffix.lower()
    if suffix in (".json", ".jsonl"):
        records, schema = parse_billing_records_from_json(content, max_records=max_records)
    elif suffix in (".csv", ".tsv"):
        records, schema = parse_billing_records_from_csv(content, max_records=max_records)
    else:
        # Fallback: attempt JSON first, then CSV
        try:
            records, schema = parse_billing_records_from_json(content, max_records=max_records)
        except Exception:
            records, schema = parse_billing_records_from_csv(content, max_records=max_records)

    logger.debug(
        f"Successfully loaded {len(records)} billing records from {resolved_path} (schema: {schema})"
    )
    return records, schema


def get_synthetic_billing_dataset() -> List[BillingRecord]:
    """Return a rich offline benchmark dataset of multi-cloud billing records."""
    base_date = datetime.datetime(2026, 11, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    records: List[BillingRecord] = []

    # 1. AWS Services
    records.append(
        BillingRecord(
            record_id="REC-AWS-EC2-01",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonEC2",
            resource_id="i-0abcd1234ef567890",
            resource_name="prod-api-worker-01",
            region="us-east-1",
            usage_type="BoxUsage:t3.xlarge",
            usage_quantity=720.0,
            usage_unit="Hrs",
            cost=120.50,
            currency="USD",
            start_time=base_date,
            end_time=base_date + datetime.timedelta(days=30),
            tags={"environment": "production", "team": "platform"},
        )
    )
    records.append(
        BillingRecord(
            record_id="REC-AWS-S3-02",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonS3",
            resource_id="arn:aws:s3:::prod-app-backups",
            resource_name="prod-app-backups",
            region="us-east-1",
            usage_type="TimedStorage-ByteHrs",
            usage_quantity=15000.0,
            usage_unit="GB-Mo",
            cost=345.00,
            currency="USD",
            start_time=base_date,
            end_time=base_date + datetime.timedelta(days=30),
            tags={"environment": "production", "type": "backup"},
        )
    )
    records.append(
        BillingRecord(
            record_id="REC-AWS-RDS-03",
            provider=CloudProvider.AWS,
            account_id="123456789012",
            service_name="AmazonRDS",
            resource_id="db-prod-postgres-primary",
            resource_name="db-prod-postgres-primary",
            region="us-east-1",
            usage_type="InstanceUsage:db.r5.2xlarge",
            usage_quantity=720.0,
            usage_unit="Hrs",
            cost=580.00,
            currency="USD",
            start_time=base_date,
            end_time=base_date + datetime.timedelta(days=30),
            tags={"environment": "production", "tier": "database"},
        )
    )

    # 2. Azure Services
    records.append(
        BillingRecord(
            record_id="REC-AZURE-VM-04",
            provider=CloudProvider.AZURE,
            account_id="sub-prod-9876",
            service_name="Virtual Machines",
            resource_id="/subscriptions/sub-prod-9876/resourceGroups/prod-rg/providers/Microsoft.Compute/virtualMachines/vm-payment-01",
            resource_name="vm-payment-01",
            region="eastus",
            usage_type="Standard_D4s_v5",
            usage_quantity=720.0,
            usage_unit="Hours",
            cost=142.35,
            currency="USD",
            start_time=base_date,
            end_time=base_date + datetime.timedelta(days=30),
            tags={"environment": "production"},
        )
    )
    records.append(
        BillingRecord(
            record_id="REC-AZURE-BLOB-05",
            provider=CloudProvider.AZURE,
            account_id="sub-prod-9876",
            service_name="Storage",
            resource_id="/subscriptions/sub-prod-9876/resourceGroups/prod-rg/providers/Microsoft.Storage/storageAccounts/stprodlogs",
            resource_name="stprodlogs",
            region="eastus",
            usage_type="Hot LRS Data Stored",
            usage_quantity=5000.0,
            usage_unit="GB",
            cost=105.00,
            currency="USD",
            start_time=base_date,
            end_time=base_date + datetime.timedelta(days=30),
            tags={"environment": "production"},
        )
    )

    # 3. GCP Services
    records.append(
        BillingRecord(
            record_id="REC-GCP-GCE-06",
            provider=CloudProvider.GCP,
            account_id="gcp-analytics-project",
            service_name="Compute Engine",
            resource_id="analytics-worker-e2-standard-4",
            resource_name="analytics-worker-e2-standard-4",
            region="us-central1",
            usage_type="N1 Predefined Instance Core",
            usage_quantity=720.0,
            usage_unit="Hours",
            cost=118.40,
            currency="USD",
            start_time=base_date,
            end_time=base_date + datetime.timedelta(days=30),
            tags={"environment": "data"},
        )
    )
    records.append(
        BillingRecord(
            record_id="REC-GCP-BQ-07",
            provider=CloudProvider.GCP,
            account_id="gcp-analytics-project",
            service_name="BigQuery",
            resource_id="gcp-analytics-project:telemetry_dataset",
            resource_name="telemetry_dataset",
            region="us-central1",
            usage_type="Analysis Queries",
            usage_quantity=25.0,
            usage_unit="TB",
            cost=125.00,
            currency="USD",
            start_time=base_date,
            end_time=base_date + datetime.timedelta(days=30),
            tags={"environment": "data"},
        )
    )

    return records
