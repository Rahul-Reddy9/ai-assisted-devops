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
from devops_ai.finops.models import (
    BillingRecord,
    CloudProvider,
    CloudResource,
    ResourceCategory,
    ResourceUtilization,
)

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


def load_cloud_resources(
    path: Union[str, Path],
    allowed_root: Optional[Path] = None,
    max_records: int = 100000,
    max_file_size_bytes: Optional[int] = None,
) -> List[CloudResource]:
    """Safely load and validate cloud resource inventory from a CSV or JSON file."""
    file_path = Path(path)
    try:
        resolved_path = file_path.resolve()
    except Exception as e:
        raise BillingDataError(f"Invalid filesystem path: {str(e)}", source=str(path)) from e

    if not resolved_path.exists():
        raise BillingDataError(
            f"Resource inventory file does not exist: '{resolved_path}'", source=str(path)
        )
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

    file_size = resolved_path.stat().st_size
    effective_max_size = (
        max_file_size_bytes if max_file_size_bytes is not None else MAX_BILLING_FILE_SIZE_BYTES
    )
    if file_size > effective_max_size:
        raise BillingDataError(
            f"File size ({file_size} bytes) exceeds maximum limit ({effective_max_size} bytes)",
            source=str(path),
        )

    try:
        content = resolved_path.read_text(encoding="utf-8")
    except Exception as e:
        raise BillingDataError(f"Failed to read file: {str(e)}", source=str(path)) from e

    if not content.strip():
        raise BillingDataError("Resource inventory file is empty", source=str(path))

    resources: List[CloudResource] = []
    suffix = resolved_path.suffix.lower()

    if suffix in (".json", ".jsonl"):
        try:
            data = json.loads(content)
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise BillingDataError(f"Malformed JSON: {sanitized}", source=str(path)) from e

        items = data if isinstance(data, list) else data.get("resources", [data])
        for idx, item in enumerate(items, start=1):
            if len(resources) >= max_records:
                break
            if not isinstance(item, dict):
                raise BillingDataError(f"Item {idx} is not an object", source=str(path))
            try:
                raw_prov = str(item.get("provider", "custom")).lower()
                prov = (
                    CloudProvider(raw_prov)
                    if raw_prov in [p.value for p in CloudProvider]
                    else CloudProvider.CUSTOM
                )
                raw_cat = str(item.get("category", "other")).lower()
                cat = (
                    ResourceCategory(raw_cat)
                    if raw_cat in [c.value for c in ResourceCategory]
                    else ResourceCategory.OTHER
                )

                norm_item = {
                    "resource_id": str(item["resource_id"]),
                    "resource_name": item.get("resource_name"),
                    "provider": prov,
                    "account_id": str(item.get("account_id", "default-account")),
                    "service_name": str(item.get("service_name", "General Service")),
                    "category": cat,
                    "region": item.get("region"),
                    "monthly_cost": float(item.get("monthly_cost", 0.0)),
                    "currency": str(item.get("currency", "USD")).strip().upper(),
                    "specs": item.get("specs", {}) if isinstance(item.get("specs"), dict) else {},
                    "tags": item.get("tags", {}) if isinstance(item.get("tags"), dict) else {},
                }
                resources.append(CloudResource.model_validate(norm_item))
            except Exception as e:
                sanitized = scrub_secrets(str(e))
                raise BillingDataError(
                    f"Resource {idx} validation failed: {sanitized}", source=str(path)
                ) from e
    else:
        # CSV parsing
        reader = csv.DictReader(content.splitlines())
        if not reader.fieldnames:
            raise BillingDataError("CSV contains no header row", source=str(path))

        for row_idx, row in enumerate(reader, start=1):
            if len(resources) >= max_records:
                break
            if not any(row.values()):
                continue
            try:
                raw_prov = str(row.get("provider", "custom")).lower()
                prov = (
                    CloudProvider(raw_prov)
                    if raw_prov in [p.value for p in CloudProvider]
                    else CloudProvider.CUSTOM
                )
                raw_cat = str(row.get("category", "other")).lower()
                cat = (
                    ResourceCategory(raw_cat)
                    if raw_cat in [c.value for c in ResourceCategory]
                    else ResourceCategory.OTHER
                )

                specs = {}
                raw_specs = row.get("specs")
                if raw_specs and raw_specs.strip().startswith("{"):
                    try:
                        specs = json.loads(raw_specs)
                    except Exception:
                        pass

                tags = {}
                raw_tags = row.get("tags")
                if raw_tags and raw_tags.strip().startswith("{"):
                    try:
                        tags = json.loads(raw_tags)
                    except Exception:
                        pass

                norm_row = {
                    "resource_id": str(row.get("resource_id", f"res-{row_idx}")),
                    "resource_name": row.get("resource_name") or None,
                    "provider": prov,
                    "account_id": str(row.get("account_id", "default-account")),
                    "service_name": str(row.get("service_name", "General Service")),
                    "category": cat,
                    "region": row.get("region") or None,
                    "monthly_cost": float(row.get("monthly_cost", 0.0)),
                    "currency": str(row.get("currency", "USD")).strip().upper(),
                    "specs": specs,
                    "tags": tags,
                }
                resources.append(CloudResource.model_validate(norm_row))
            except Exception as e:
                sanitized = scrub_secrets(str(e))
                raise BillingDataError(
                    f"Row {row_idx} validation failed: {sanitized}", source=str(path)
                ) from e

    if not resources:
        raise BillingDataError("No valid cloud resources found in file", source=str(path))
    return resources


def load_resource_utilization(
    path: Union[str, Path],
    allowed_root: Optional[Path] = None,
    max_records: int = 100000,
    max_file_size_bytes: Optional[int] = None,
) -> List[ResourceUtilization]:
    """Safely load and validate resource utilization metrics from a CSV or JSON file."""
    file_path = Path(path)
    try:
        resolved_path = file_path.resolve()
    except Exception as e:
        raise BillingDataError(f"Invalid filesystem path: {str(e)}", source=str(path)) from e

    if not resolved_path.exists():
        raise BillingDataError(
            f"Utilization file does not exist: '{resolved_path}'", source=str(path)
        )
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

    file_size = resolved_path.stat().st_size
    effective_max_size = (
        max_file_size_bytes if max_file_size_bytes is not None else MAX_BILLING_FILE_SIZE_BYTES
    )
    if file_size > effective_max_size:
        raise BillingDataError(
            f"File size ({file_size} bytes) exceeds maximum limit ({effective_max_size} bytes)",
            source=str(path),
        )

    try:
        content = resolved_path.read_text(encoding="utf-8")
    except Exception as e:
        raise BillingDataError(f"Failed to read file: {str(e)}", source=str(path)) from e

    if not content.strip():
        raise BillingDataError("Utilization file is empty", source=str(path))

    records: List[ResourceUtilization] = []
    suffix = resolved_path.suffix.lower()

    def _parse_bool(val: Any) -> Optional[bool]:
        if val is None or val == "":
            return None
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return None

    def _parse_float_opt(val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    if suffix in (".json", ".jsonl"):
        try:
            data = json.loads(content)
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise BillingDataError(f"Malformed JSON: {sanitized}", source=str(path)) from e

        items = data if isinstance(data, list) else data.get("utilization", [data])
        for idx, item in enumerate(items, start=1):
            if len(records) >= max_records:
                break
            if not isinstance(item, dict):
                raise BillingDataError(f"Item {idx} is not an object", source=str(path))
            try:
                ts_raw = item.get("timestamp", datetime.datetime.now(datetime.timezone.utc))
                ts = _parse_iso_or_common_date(ts_raw)
                norm_util = {
                    "resource_id": str(item["resource_id"]),
                    "service_name": str(item.get("service_name", "General Service")),
                    "timestamp": ts,
                    "avg_cpu_pct": _parse_float_opt(item.get("avg_cpu_pct")),
                    "peak_cpu_pct": _parse_float_opt(item.get("peak_cpu_pct")),
                    "avg_memory_pct": _parse_float_opt(item.get("avg_memory_pct")),
                    "peak_memory_pct": _parse_float_opt(item.get("peak_memory_pct")),
                    "disk_read_bytes_per_sec": _parse_float_opt(
                        item.get("disk_read_bytes_per_sec")
                    ),
                    "disk_write_bytes_per_sec": _parse_float_opt(
                        item.get("disk_write_bytes_per_sec")
                    ),
                    "network_in_bytes": _parse_float_opt(item.get("network_in_bytes")),
                    "network_out_bytes": _parse_float_opt(item.get("network_out_bytes")),
                    "is_attached": _parse_bool(item.get("is_attached")),
                }
                records.append(ResourceUtilization.model_validate(norm_util))
            except Exception as e:
                sanitized = scrub_secrets(str(e))
                raise BillingDataError(
                    f"Utilization item {idx} validation failed: {sanitized}", source=str(path)
                ) from e
    else:
        # CSV parsing
        reader = csv.DictReader(content.splitlines())
        if not reader.fieldnames:
            raise BillingDataError("CSV contains no header row", source=str(path))

        for row_idx, row in enumerate(reader, start=1):
            if len(records) >= max_records:
                break
            if not any(row.values()):
                continue
            try:
                ts_raw = row.get("timestamp") or datetime.datetime.now(datetime.timezone.utc)
                ts = _parse_iso_or_common_date(ts_raw)
                norm_row = {
                    "resource_id": str(row.get("resource_id", f"res-{row_idx}")),
                    "service_name": str(row.get("service_name", "General Service")),
                    "timestamp": ts,
                    "avg_cpu_pct": _parse_float_opt(row.get("avg_cpu_pct")),
                    "peak_cpu_pct": _parse_float_opt(row.get("peak_cpu_pct")),
                    "avg_memory_pct": _parse_float_opt(row.get("avg_memory_pct")),
                    "peak_memory_pct": _parse_float_opt(row.get("peak_memory_pct")),
                    "disk_read_bytes_per_sec": _parse_float_opt(row.get("disk_read_bytes_per_sec")),
                    "disk_write_bytes_per_sec": _parse_float_opt(
                        row.get("disk_write_bytes_per_sec")
                    ),
                    "network_in_bytes": _parse_float_opt(row.get("network_in_bytes")),
                    "network_out_bytes": _parse_float_opt(row.get("network_out_bytes")),
                    "is_attached": _parse_bool(row.get("is_attached")),
                }
                records.append(ResourceUtilization.model_validate(norm_row))
            except Exception as e:
                sanitized = scrub_secrets(str(e))
                raise BillingDataError(
                    f"Utilization row {row_idx} validation failed: {sanitized}", source=str(path)
                ) from e

    if not records:
        raise BillingDataError("No valid utilization records found in file", source=str(path))
    return records


def get_synthetic_resource_inventory() -> Tuple[
    List[CloudResource], Dict[str, List[ResourceUtilization]]
]:
    """Return synthetic cloud resource inventory and telemetry covering all 5 optimization scenarios."""
    now = datetime.datetime.now(datetime.timezone.utc)
    resources: List[CloudResource] = []
    utilization_map: Dict[str, List[ResourceUtilization]] = {}

    # 1. Idle Compute Instance (AWS EC2)
    res_idle = CloudResource(
        resource_id="i-0idleworker12345",
        resource_name="dev-test-idle-worker",
        provider=CloudProvider.AWS,
        account_id="123456789012",
        service_name="AmazonEC2",
        category=ResourceCategory.COMPUTE,
        region="us-east-1",
        monthly_cost=145.00,
        currency="USD",
        specs={"instance_type": "m5.large", "vcpu": 2, "memory_gb": 8},
        tags={"environment": "development", "owner": "qa-team"},
    )
    resources.append(res_idle)
    utilization_map[res_idle.resource_id] = [
        ResourceUtilization(
            resource_id=res_idle.resource_id,
            service_name="AmazonEC2",
            timestamp=now - datetime.timedelta(days=i),
            avg_cpu_pct=0.6,
            peak_cpu_pct=1.8,
            avg_memory_pct=4.2,
            peak_memory_pct=6.5,
            network_in_bytes=0.0,
            network_out_bytes=0.0,
            is_attached=True,
        )
        for i in range(14)
    ]

    # 2. Underutilized Compute Instance (Azure VM)
    res_underutilized = CloudResource(
        resource_id="/subscriptions/sub-1/resourceGroups/rg-app/providers/Microsoft.Compute/virtualMachines/vm-underutilized-web-02",
        resource_name="vm-underutilized-web-02",
        provider=CloudProvider.AZURE,
        account_id="sub-1",
        service_name="Virtual Machines",
        category=ResourceCategory.COMPUTE,
        region="eastus",
        monthly_cost=275.00,
        currency="USD",
        specs={"instance_type": "Standard_D8s_v5", "vcpu": 8, "memory_gb": 32},
        tags={"environment": "production", "tier": "web"},
    )
    resources.append(res_underutilized)
    utilization_map[res_underutilized.resource_id] = [
        ResourceUtilization(
            resource_id=res_underutilized.resource_id,
            service_name="Virtual Machines",
            timestamp=now - datetime.timedelta(days=i),
            avg_cpu_pct=8.4,
            peak_cpu_pct=24.0,
            avg_memory_pct=18.5,
            peak_memory_pct=29.0,
            network_in_bytes=500000.0,
            network_out_bytes=750000.0,
            is_attached=True,
        )
        for i in range(14)
    ]

    # 3. Unattached Storage Volume (AWS EBS)
    res_storage = CloudResource(
        resource_id="vol-0unattached8899",
        resource_name="orphaned-backup-disk",
        provider=CloudProvider.AWS,
        account_id="123456789012",
        service_name="AmazonEBS",
        category=ResourceCategory.STORAGE,
        region="us-east-1",
        monthly_cost=80.00,
        currency="USD",
        specs={"volume_type": "gp3", "size_gb": 1000},
        tags={"snapshot_date": "2026-10-01"},
    )
    resources.append(res_storage)
    utilization_map[res_storage.resource_id] = [
        ResourceUtilization(
            resource_id=res_storage.resource_id,
            service_name="AmazonEBS",
            timestamp=now - datetime.timedelta(days=i),
            disk_read_bytes_per_sec=0.0,
            disk_write_bytes_per_sec=0.0,
            is_attached=False,
        )
        for i in range(14)
    ]

    # 4. Over-provisioned Database (AWS RDS)
    res_db = CloudResource(
        resource_id="db-overprovisioned-analytics-04",
        resource_name="db-analytics-stage",
        provider=CloudProvider.AWS,
        account_id="123456789012",
        service_name="AmazonRDS",
        category=ResourceCategory.DATABASE,
        region="us-east-1",
        monthly_cost=580.00,
        currency="USD",
        specs={"instance_class": "db.r5.2xlarge", "vcpu": 8, "memory_gb": 64},
        tags={"environment": "staging"},
    )
    resources.append(res_db)
    utilization_map[res_db.resource_id] = [
        ResourceUtilization(
            resource_id=res_db.resource_id,
            service_name="AmazonRDS",
            timestamp=now - datetime.timedelta(days=i),
            avg_cpu_pct=5.5,
            peak_cpu_pct=14.2,
            avg_memory_pct=12.0,
            peak_memory_pct=22.0,
            is_attached=True,
        )
        for i in range(14)
    ]

    # 5. Orphaned Static Public IP (AWS Elastic IP)
    res_ip = CloudResource(
        resource_id="eipalloc-0123456789abcdef0",
        resource_name="eip-unused-legacy",
        provider=CloudProvider.AWS,
        account_id="123456789012",
        service_name="AmazonEC2",
        category=ResourceCategory.NETWORK,
        region="us-east-1",
        monthly_cost=3.60,
        currency="USD",
        specs={"public_ip": "54.210.88.99", "allocation_id": "eipalloc-0123456789abcdef0"},
        tags={"created_by": "terraform-legacy"},
    )
    resources.append(res_ip)
    utilization_map[res_ip.resource_id] = [
        ResourceUtilization(
            resource_id=res_ip.resource_id,
            service_name="AmazonEC2",
            timestamp=now - datetime.timedelta(days=i),
            network_in_bytes=0.0,
            network_out_bytes=0.0,
            is_attached=False,
        )
        for i in range(14)
    ]

    # 6. Healthy Active Compute Resource (No optimization recommended - zero false positives)
    res_healthy = CloudResource(
        resource_id="prod-api-active-worker-06",
        resource_name="prod-api-worker-06",
        provider=CloudProvider.GCP,
        account_id="gcp-prod-infra",
        service_name="Compute Engine",
        category=ResourceCategory.COMPUTE,
        region="us-central1",
        monthly_cost=210.00,
        currency="USD",
        specs={"machine_type": "e2-standard-4", "vcpu": 4, "memory_gb": 16},
        tags={"environment": "production"},
    )
    resources.append(res_healthy)
    utilization_map[res_healthy.resource_id] = [
        ResourceUtilization(
            resource_id=res_healthy.resource_id,
            service_name="Compute Engine",
            timestamp=now - datetime.timedelta(days=i),
            avg_cpu_pct=52.0,
            peak_cpu_pct=84.5,
            avg_memory_pct=65.0,
            peak_memory_pct=78.0,
            network_in_bytes=4500000.0,
            network_out_bytes=6200000.0,
            is_attached=True,
        )
        for i in range(14)
    ]

    return resources, utilization_map


def get_synthetic_anomaly_and_forecast_dataset() -> List[BillingRecord]:
    """Return a rich offline daily benchmark dataset containing stable spend, spikes, and growth trends."""
    base_date = datetime.datetime(2026, 10, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    records: List[BillingRecord] = []

    # 45 days of simulated multi-cloud daily expenditures
    for day in range(45):
        dt = base_date + datetime.timedelta(days=day)
        date_str = dt.strftime("%Y%m%d")

        # 1. AWS EC2: Stable baseline ~$100/day, with sudden runaway batch surge on Days 25, 26, 27 ($380-$440/day)
        if 25 <= day <= 27:
            ec2_cost = 410.0 + (day % 3) * 15.0
            ec2_res = "i-batch-worker-runaway"
            ec2_name = "batch-worker-runaway-surge"
        else:
            ec2_cost = 98.0 + (day % 5) * 1.5
            ec2_res = "i-0prodworker1234"
            ec2_name = "prod-api-worker"

        records.append(
            BillingRecord(
                record_id=f"REC-AWS-EC2-{date_str}",
                provider=CloudProvider.AWS,
                account_id="123456789012",
                service_name="AmazonEC2",
                resource_id=ec2_res,
                resource_name=ec2_name,
                region="us-east-1",
                usage_type="BoxUsage:m5.xlarge",
                usage_quantity=24.0,
                usage_unit="Hrs",
                cost=round(ec2_cost, 2),
                currency="USD",
                start_time=dt,
                end_time=dt + datetime.timedelta(days=1),
                tags={"environment": "production"},
            )
        )

        # 2. AWS S3: Stable baseline ~$18/day, with egress spike on Day 34 ($195)
        if day == 34:
            s3_cost = 195.50
            s3_res = "arn:aws:s3:::data-warehouse-egress"
        else:
            s3_cost = 18.0 + (day % 4) * 0.5
            s3_res = "arn:aws:s3:::app-backups"

        records.append(
            BillingRecord(
                record_id=f"REC-AWS-S3-{date_str}",
                provider=CloudProvider.AWS,
                account_id="123456789012",
                service_name="AmazonS3",
                resource_id=s3_res,
                region="us-east-1",
                usage_type="TimedStorage-ByteHrs",
                usage_quantity=500.0,
                usage_unit="GB-Mo",
                cost=round(s3_cost, 2),
                currency="USD",
                start_time=dt,
                end_time=dt + datetime.timedelta(days=1),
                tags={"tier": "storage"},
            )
        )

        # 3. Azure VMs: Steady state ~$75/day
        records.append(
            BillingRecord(
                record_id=f"REC-AZURE-VM-{date_str}",
                provider=CloudProvider.AZURE,
                account_id="sub-prod-9876",
                service_name="Virtual Machines",
                resource_id="/subscriptions/sub-prod-9876/resourceGroups/app-rg/providers/Microsoft.Compute/virtualMachines/vm-frontend-01",
                region="eastus",
                usage_type="Standard_D4s_v5",
                usage_quantity=24.0,
                usage_unit="Hours",
                cost=round(75.0 + (day % 3) * 0.8, 2),
                currency="USD",
                start_time=dt,
                end_time=dt + datetime.timedelta(days=1),
                tags={"environment": "production"},
            )
        )

        # 4. GCP Cloud SQL: Linear steady growth ($20 on Day 0 -> $55 on Day 44)
        gcp_cost = 20.0 + (day * 0.8)
        records.append(
            BillingRecord(
                record_id=f"REC-GCP-SQL-{date_str}",
                provider=CloudProvider.GCP,
                account_id="gcp-analytics-prod",
                service_name="Cloud SQL",
                resource_id="projects/gcp-analytics-prod/instances/analytics-db",
                region="us-central1",
                usage_type="DBInstance:db-custom-4-16",
                usage_quantity=24.0,
                usage_unit="Hours",
                cost=round(gcp_cost, 2),
                currency="USD",
                start_time=dt,
                end_time=dt + datetime.timedelta(days=1),
                tags={"workload": "analytics"},
            )
        )

        # 5. Promotional credit adjustment on Day 14 (-$50.00)
        if day == 14:
            records.append(
                BillingRecord(
                    record_id="REC-AWS-CREDIT-20261015",
                    provider=CloudProvider.AWS,
                    account_id="123456789012",
                    service_name="AmazonEC2",
                    usage_type="CreditAdjustment",
                    cost=-50.00,
                    currency="USD",
                    start_time=dt,
                    end_time=dt + datetime.timedelta(days=1),
                    tags={"type": "promotional-credit"},
                )
            )

    return records
