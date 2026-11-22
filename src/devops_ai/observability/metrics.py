# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Prometheus metrics registry, collectors, and low-cardinality telemetry."""

import threading
import time
from typing import Optional
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from devops_ai.core.logging import get_logger

logger = get_logger("observability.metrics")

# Dedicated registry to isolate application metrics from third-party libraries
METRICS_REGISTRY = CollectorRegistry(auto_describe=True)

# Low-cardinality metric definitions (strict bounded labels)
LLM_GENERATION_REQUESTS = Counter(
    "devops_ai_generation_requests_total",
    "Total infrastructure manifest generation requests",
    ["generator", "provider", "status"],
    registry=METRICS_REGISTRY,
)

LLM_GENERATION_DURATION = Histogram(
    "devops_ai_generation_duration_seconds",
    "End-to-end latency of manifest generation operations",
    ["generator", "provider"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=METRICS_REGISTRY,
)

VALIDATION_WARNINGS = Counter(
    "devops_ai_validation_warnings_total",
    "Total security or heuristic warnings detected during validation",
    ["generator", "warning_type"],
    registry=METRICS_REGISTRY,
)

VALIDATION_ERRORS = Counter(
    "devops_ai_validation_errors_total",
    "Total manifest structural or syntax validation errors",
    ["generator", "error_code"],
    registry=METRICS_REGISTRY,
)

PROVIDER_HEALTH_GAUGE = Gauge(
    "devops_ai_provider_health_status",
    "Current health status of LLM providers (1 = healthy, 0 = unhealthy)",
    ["provider"],
    registry=METRICS_REGISTRY,
)

PROVIDER_PROBE_LATENCY = Gauge(
    "devops_ai_provider_probe_duration_seconds",
    "Round-trip latency of the most recent provider health probe",
    ["provider"],
    registry=METRICS_REGISTRY,
)

DOCKER_AVAILABLE_GAUGE = Gauge(
    "devops_ai_docker_available",
    "Docker daemon connectivity state (1 = reachable, 0 = unavailable)",
    registry=METRICS_REGISTRY,
)

K8S_AVAILABLE_GAUGE = Gauge(
    "devops_ai_k8s_available",
    "Kubernetes cluster connectivity state (1 = reachable, 0 = unavailable)",
    registry=METRICS_REGISTRY,
)

LOG_ANOMALIES_COUNTER = Counter(
    "devops_ai_log_anomalies_detected_total",
    "Total operational log anomalies identified",
    ["detection_type"],
    registry=METRICS_REGISTRY,
)

SYSTEM_CPU_GAUGE = Gauge(
    "devops_ai_system_cpu_usage_percent",
    "Host system CPU utilization percentage",
    registry=METRICS_REGISTRY,
)

SYSTEM_MEMORY_GAUGE = Gauge(
    "devops_ai_system_memory_usage_bytes",
    "Host system physical memory utilization in bytes",
    registry=METRICS_REGISTRY,
)


class MetricsCache:
    """Thread-safe cached metrics exposition to avoid blocking on Prometheus scrapes."""

    def __init__(self, ttl_seconds: float = 2.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._cached_bytes: bytes = b""
        self._last_collected: float = 0.0
        self._lock = threading.Lock()

    def get_exposition(self, force_refresh: bool = False) -> bytes:
        now = time.monotonic()
        with self._lock:
            if (
                force_refresh
                or not self._cached_bytes
                or (now - self._last_collected) >= self.ttl_seconds
            ):
                update_system_metrics()
                self._cached_bytes = generate_latest(METRICS_REGISTRY)
                self._last_collected = now
            return self._cached_bytes


_METRICS_CACHE = MetricsCache(ttl_seconds=2.0)


def update_system_metrics() -> None:
    """Collect host CPU and memory metrics safely using psutil."""
    try:
        import psutil  # type: ignore[import-untyped]

        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        SYSTEM_CPU_GAUGE.set(float(cpu_pct))
        SYSTEM_MEMORY_GAUGE.set(float(mem.used))
    except Exception as e:
        logger.debug(f"Failed to sample system metrics: {e}")


def record_generation_request(
    generator: str,
    provider: str,
    status: str,
    duration_seconds: Optional[float] = None,
) -> None:
    """Record manifest generation metrics with strictly bounded labels."""
    safe_gen = generator.lower() if generator in ("dockerfile", "k8s") else "unknown"
    safe_prov = provider.lower() if provider in ("ollama", "gemini") else "unknown"
    safe_status = status.lower() if status in ("success", "failure", "fallback") else "unknown"

    LLM_GENERATION_REQUESTS.labels(
        generator=safe_gen,
        provider=safe_prov,
        status=safe_status,
    ).inc()

    if duration_seconds is not None and duration_seconds >= 0:
        LLM_GENERATION_DURATION.labels(
            generator=safe_gen,
            provider=safe_prov,
        ).observe(duration_seconds)


def record_validation_warning(generator: str, warning_type: str) -> None:
    """Record heuristic warning with low-cardinality category."""
    safe_gen = generator.lower() if generator in ("dockerfile", "k8s") else "unknown"
    safe_type = (
        warning_type.lower() if warning_type in ("security", "syntax", "heuristic") else "heuristic"
    )
    VALIDATION_WARNINGS.labels(generator=safe_gen, warning_type=safe_type).inc()


def record_validation_error(generator: str, error_code: str) -> None:
    """Record fatal validation error with bounded error code."""
    safe_gen = generator.lower() if generator in ("dockerfile", "k8s") else "unknown"
    VALIDATION_ERRORS.labels(generator=safe_gen, error_code=error_code).inc()


def record_provider_health(provider: str, is_healthy: bool, probe_duration: float) -> None:
    """Update provider health gauge and probe latency."""
    safe_prov = provider.lower() if provider in ("ollama", "gemini") else "unknown"
    PROVIDER_HEALTH_GAUGE.labels(provider=safe_prov).set(1.0 if is_healthy else 0.0)
    PROVIDER_PROBE_LATENCY.labels(provider=safe_prov).set(max(0.0, probe_duration))


def record_log_anomaly(detection_type: str) -> None:
    """Increment counter for detected log anomalies."""
    safe_type = (
        detection_type.lower()
        if detection_type in ("burst", "ml_isolation_forest", "statistical")
        else "burst"
    )
    LOG_ANOMALIES_COUNTER.labels(detection_type=safe_type).inc()


def record_docker_status(is_available: bool) -> None:
    """Update Docker daemon availability gauge (1.0 = available, 0.0 = unavailable)."""
    DOCKER_AVAILABLE_GAUGE.set(1.0 if is_available else 0.0)


def record_k8s_status(is_available: bool) -> None:
    """Update Kubernetes cluster availability gauge (1.0 = available, 0.0 = unavailable)."""
    K8S_AVAILABLE_GAUGE.set(1.0 if is_available else 0.0)


def get_prometheus_exposition(force_refresh: bool = False) -> bytes:
    """Retrieve Prometheus metrics payload in standard exposition format."""
    return _METRICS_CACHE.get_exposition(force_refresh=force_refresh)
