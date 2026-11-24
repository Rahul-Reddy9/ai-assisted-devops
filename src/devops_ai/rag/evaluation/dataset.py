# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Evaluation dataset loader, validator, and built-in synthetic DevOps evaluation suite."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from devops_ai.core.exceptions import DatasetValidationError
from devops_ai.core.logging import get_logger, scrub_secrets
from devops_ai.rag.evaluation.models import EvaluationDataset

logger = get_logger("rag.evaluation.dataset")

MAX_DATASET_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


# Built-in synthetic evaluation benchmark representing core production incidents and runbooks
_BUILTIN_DATASET_DATA: Dict[str, Any] = {
    "dataset_id": "builtin-devops-v1",
    "name": "Standard DevOps Runbook & Incident Benchmark",
    "description": "Standard benchmark suite covering Kubernetes, databases, caching, storage, and networking incidents.",
    "version": "1.0.0",
    "queries": [
        {
            "query_id": "Q-K8S-OOM-01",
            "query": "Kubernetes pod CrashLoopBackOff OOMKilled exit code 137 container memory limit",
            "expected_document_ids": ["DOC-K8S-RUNBOOK-01"],
            "expected_chunk_ids": ["DOC-K8S-RUNBOOK-01-C0"],
            "graded_relevance": {"DOC-K8S-RUNBOOK-01": 3.0, "DOC-DOCKER-GUIDE-02": 1.0},
            "reference_answer": "Increase container resources.limits.memory in the deployment manifest [DOC-K8S-RUNBOOK-01]. Check node kernel dmesg for OOM killer invocations.",
            "expected_facts": [
                "Increase container resources.limits.memory",
                "Exit code 137 indicates SIGKILL by OOM killer",
            ],
            "metadata": {"category": "kubernetes", "severity": "high"},
        },
        {
            "query_id": "Q-PG-CONN-02",
            "query": "PostgreSQL FATAL remaining connection slots are reserved for non-replication superuser connections",
            "expected_document_ids": ["DOC-PG-OPS-03"],
            "expected_chunk_ids": ["DOC-PG-OPS-03-C0"],
            "graded_relevance": {"DOC-PG-OPS-03": 3.0},
            "reference_answer": "Connection exhaustion reached. Tune max_connections in postgresql.conf and deploy PgBouncer connection pooling [DOC-PG-OPS-03].",
            "expected_facts": [
                "Exhaustion of max_connections pool",
                "Implement PgBouncer connection pooling",
            ],
            "metadata": {"category": "database", "severity": "critical"},
        },
        {
            "query_id": "Q-REDIS-MEM-03",
            "query": "Redis high latency spike instantaneous ops per second drop memory fragmentation ratio",
            "expected_document_ids": ["DOC-REDIS-OPS-04"],
            "expected_chunk_ids": ["DOC-REDIS-OPS-04-C0"],
            "graded_relevance": {"DOC-REDIS-OPS-04": 3.0},
            "reference_answer": "High fragmentation ratio (> 1.5) triggers swapping. Enable activedefrag in redis.conf and inspect slowlog [DOC-REDIS-OPS-04].",
            "expected_facts": [
                "Enable activedefrag yes",
                "Inspect slowlog get 25",
            ],
            "metadata": {"category": "cache", "severity": "medium"},
        },
        {
            "query_id": "Q-DISK-INODE-04",
            "query": "No space left on device error but df -h shows available disk space inode exhaustion",
            "expected_document_ids": ["DOC-LINUX-STORAGE-05"],
            "expected_chunk_ids": ["DOC-LINUX-STORAGE-05-C0"],
            "graded_relevance": {"DOC-LINUX-STORAGE-05": 3.0},
            "reference_answer": "Run df -i to inspect inode usage. Find directories with millions of small files using find and xargs rm [DOC-LINUX-STORAGE-05].",
            "expected_facts": [
                "Run df -i to check 100% inode saturation",
                "Clean up abandoned session or cache directories",
            ],
            "metadata": {"category": "linux", "severity": "high"},
        },
        {
            "query_id": "Q-NGINX-502-05",
            "query": "NGINX 502 Bad Gateway connect failed 111 Connection refused upstream server",
            "expected_document_ids": ["DOC-NGINX-REVERSE-PROXY-06"],
            "expected_chunk_ids": ["DOC-NGINX-REVERSE-PROXY-06-C0"],
            "graded_relevance": {"DOC-NGINX-REVERSE-PROXY-06": 3.0},
            "reference_answer": "The backend application server is down or not listening on the specified upstream socket [DOC-NGINX-REVERSE-PROXY-06]. Verify systemctl status of backend service.",
            "expected_facts": [
                "Upstream process is terminated or port is closed",
                "Verify upstream socket and port configuration",
            ],
            "metadata": {"category": "network", "severity": "high"},
        },
        {
            "query_id": "Q-KAFKA-REBALANCE-06",
            "query": "Kafka consumer group rebalance storm max poll interval ms exceeded commit failed",
            "expected_document_ids": ["DOC-KAFKA-CLUSTER-07"],
            "expected_chunk_ids": ["DOC-KAFKA-CLUSTER-07-C0"],
            "graded_relevance": {"DOC-KAFKA-CLUSTER-07": 3.0},
            "reference_answer": "Consumer batch processing exceeded max.poll.interval.ms. Reduce max.poll.records or increase interval [DOC-KAFKA-CLUSTER-07].",
            "expected_facts": [
                "Consumer processing duration exceeds timeout",
                "Tune max.poll.records or max.poll.interval.ms",
            ],
            "metadata": {"category": "messaging", "severity": "medium"},
        },
        {
            "query_id": "Q-ES-CIRCUIT-07",
            "query": "Elasticsearch CircuitBreakingException parent Data too large bytes limit exceeded",
            "expected_document_ids": ["DOC-ES-SEARCH-08"],
            "expected_chunk_ids": ["DOC-ES-SEARCH-08-C0"],
            "graded_relevance": {"DOC-ES-SEARCH-08": 3.0},
            "reference_answer": "Parent circuit breaker tripped to prevent node JVM crash [DOC-ES-SEARCH-08]. Reduce query aggregation bucket size and increase JVM heap.",
            "expected_facts": [
                "JVM heap saturation in search aggregations",
                "Adjust indices.breaker.total.use_real_memory",
            ],
            "metadata": {"category": "search", "severity": "high"},
        },
        {
            "query_id": "Q-CERT-EXPIRY-08",
            "query": "TLS certificate expired ERR_SSL_VERSION_OR_CIPHER_MISMATCH renew certbot Let's Encrypt",
            "expected_document_ids": ["DOC-SECURITY-TLS-09"],
            "expected_chunk_ids": ["DOC-SECURITY-TLS-09-C0"],
            "graded_relevance": {"DOC-SECURITY-TLS-09": 3.0},
            "reference_answer": "Execute certbot renew --force-renewal and reload NGINX or reverse proxy [DOC-SECURITY-TLS-09].",
            "expected_facts": [
                "Run certbot renew",
                "Reload reverse proxy configuration",
            ],
            "metadata": {"category": "security", "severity": "critical"},
        },
        {
            "query_id": "Q-DOCKER-FD-09",
            "query": "Docker daemon error failed to create shim task too many open files file-max limit",
            "expected_document_ids": ["DOC-DOCKER-SYSCTL-10"],
            "expected_chunk_ids": ["DOC-DOCKER-SYSCTL-10-C0"],
            "graded_relevance": {"DOC-DOCKER-SYSCTL-10": 3.0},
            "reference_answer": "System file descriptor limit exhausted. Increase fs.file-max in sysctl.conf and nofile in /etc/security/limits.conf [DOC-DOCKER-SYSCTL-10].",
            "expected_facts": [
                "Systemwide file descriptor exhaustion",
                "Increase fs.file-max and ulimit -n",
            ],
            "metadata": {"category": "docker", "severity": "high"},
        },
        {
            "query_id": "Q-PROM-FLAPPING-10",
            "query": "Prometheus alertmanager notification storm flapping alerts alert evaluation interval",
            "expected_document_ids": ["DOC-PROMETHEUS-OPS-11"],
            "expected_chunk_ids": ["DOC-PROMETHEUS-OPS-11-C0"],
            "graded_relevance": {"DOC-PROMETHEUS-OPS-11": 3.0},
            "reference_answer": "Flapping alerts caused by noisy threshold crossing. Increase the 'for' duration clause in Prometheus alerting rules [DOC-PROMETHEUS-OPS-11].",
            "expected_facts": [
                "Configure longer 'for' duration to debounce alert",
                "Tune group_wait and repeat_interval in Alertmanager",
            ],
            "metadata": {"category": "observability", "severity": "medium"},
        },
    ],
}


def get_builtin_evaluation_dataset() -> EvaluationDataset:
    """Return a pristine instance of the built-in synthetic DevOps evaluation dataset."""
    return EvaluationDataset.model_validate(_BUILTIN_DATASET_DATA)


def load_evaluation_dataset(
    path: Union[str, Path],
    allowed_root: Optional[Path] = None,
    max_queries: int = 500,
) -> EvaluationDataset:
    """Safely load and validate an evaluation dataset from a JSON or YAML file.

    Args:
        path: Path to the dataset file.
        allowed_root: Optional root directory path to restrict path traversal attacks.
        max_queries: Maximum number of queries permitted in the dataset.

    Returns:
        Validated EvaluationDataset instance.

    Raises:
        DatasetValidationError: If file does not exist, violates path boundaries,
            exceeds size limits, contains invalid schemas, or has too many queries.
    """
    file_path = Path(path)

    # 1. Path existence and security boundary check
    try:
        resolved_path = file_path.resolve()
    except Exception as e:
        raise DatasetValidationError(f"Invalid filesystem path: {str(e)}", path=str(path)) from e

    if not resolved_path.exists():
        raise DatasetValidationError(f"File does not exist: '{resolved_path}'", path=str(path))

    if not resolved_path.is_file():
        raise DatasetValidationError(
            f"Path is not a regular file: '{resolved_path}'", path=str(path)
        )

    if allowed_root is not None:
        try:
            resolved_root = allowed_root.resolve()
            if not resolved_path.is_relative_to(resolved_root):
                raise DatasetValidationError(
                    f"Path '{resolved_path}' breaches permitted root directory '{resolved_root}'",
                    path=str(path),
                )
        except AttributeError:
            # Python < 3.9 fallback if needed
            if resolved_root not in resolved_path.parents and resolved_root != resolved_path:
                raise DatasetValidationError(
                    f"Path '{resolved_path}' breaches permitted root directory '{resolved_root}'",
                    path=str(path),
                )

    # 2. File size bound check
    file_size = resolved_path.stat().st_size
    if file_size > MAX_DATASET_FILE_SIZE_BYTES:
        raise DatasetValidationError(
            f"File size ({file_size} bytes) exceeds maximum limit ({MAX_DATASET_FILE_SIZE_BYTES} bytes)",
            path=str(path),
        )

    # 3. File extension and parsing
    suffix = resolved_path.suffix.lower()
    raw_content = ""
    try:
        raw_content = resolved_path.read_text(encoding="utf-8")
    except Exception as e:
        raise DatasetValidationError(f"Failed to read file: {str(e)}", path=str(path)) from e

    if not raw_content.strip():
        raise DatasetValidationError("Dataset file is empty", path=str(path))

    raw_data: Any = None
    try:
        if suffix in (".yaml", ".yml"):
            raw_data = yaml.safe_load(raw_content)
        elif suffix == ".json":
            raw_data = json.loads(raw_content)
        else:
            # Attempt JSON first, then YAML
            try:
                raw_data = json.loads(raw_content)
            except Exception:
                raw_data = yaml.safe_load(raw_content)
    except Exception as e:
        sanitized = scrub_secrets(str(e))
        raise DatasetValidationError(
            f"Failed to parse dataset file: {sanitized}", path=str(path)
        ) from e

    if not isinstance(raw_data, dict):
        raise DatasetValidationError("Dataset root must be a dictionary/object", path=str(path))

    # 4. Pydantic schema validation
    try:
        dataset = EvaluationDataset.model_validate(raw_data)
    except Exception as e:
        sanitized = scrub_secrets(str(e))
        raise DatasetValidationError(
            f"Schema validation failure: {sanitized}", path=str(path)
        ) from e

    # 5. Semantic dataset constraints
    if not dataset.queries:
        raise DatasetValidationError("Dataset must contain at least one query", path=str(path))

    if len(dataset.queries) > max_queries:
        raise DatasetValidationError(
            f"Query count ({len(dataset.queries)}) exceeds maximum allowed ({max_queries})",
            path=str(path),
        )

    logger.debug(
        f"Loaded evaluation dataset '{dataset.dataset_id}' with {len(dataset.queries)} queries from {resolved_path}"
    )
    return dataset


def save_evaluation_dataset(dataset: EvaluationDataset, path: Union[str, Path]) -> None:
    """Save an EvaluationDataset instance to a JSON file."""
    file_path = Path(path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = dataset.model_dump_json(indent=2)
    file_path.write_text(content, encoding="utf-8")
    logger.debug(f"Saved evaluation dataset '{dataset.dataset_id}' to {file_path}")
