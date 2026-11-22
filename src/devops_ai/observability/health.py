# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Application health evaluation and liveness/readiness probes."""

import datetime
import tempfile
import time
from typing import Any, Dict, Tuple
from devops_ai import __version__
from devops_ai.core.config import get_settings
from devops_ai.core.logging import get_logger

logger = get_logger("observability.health")

_APP_START_TIME = time.monotonic()


def evaluate_liveness() -> Dict[str, Any]:
    """Liveness probe verifying process responsiveness and uptime.

    Guaranteed execution in < 5ms without external network or daemon calls.
    """
    uptime = max(0.0, time.monotonic() - _APP_START_TIME)
    return {
        "status": "healthy",
        "service": "devops-ai",
        "version": __version__,
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def evaluate_readiness() -> Tuple[int, Dict[str, Any]]:
    """Readiness probe verifying operational state of local runtime resources.

    Returns:
        Tuple of (http_status_code, payload_dict)
        - 200: Service is ready or operational with non-critical warnings (degraded).
        - 503: Service cannot accept requests due to fatal local failure (e.g. read-only disk).
    """
    checks: Dict[str, Dict[str, Any]] = {}
    is_fatal = False
    is_degraded = False

    # 1. Local filesystem temporary directory writability
    try:
        with tempfile.NamedTemporaryFile(prefix=".devops_ai_readyz_", delete=True) as tf:
            tf.write(b"health_check")
            tf.flush()
        checks["filesystem"] = {"status": "ok", "message": "Temporary storage is writable."}
    except Exception as e:
        is_fatal = True
        checks["filesystem"] = {"status": "failed", "message": f"Storage is not writable: {e}"}

    # 2. Host memory headroom
    try:
        import psutil  # type: ignore[import-untyped]

        mem = psutil.virtual_memory()
        # Warn if free memory is critically low (< 50MB)
        if mem.available < 50 * 1024 * 1024:
            is_degraded = True
            checks["memory"] = {
                "status": "warning",
                "message": f"Low memory headroom: {mem.available // (1024 * 1024)}MB available.",
            }
        else:
            checks["memory"] = {
                "status": "ok",
                "message": f"{mem.available // (1024 * 1024)}MB memory available.",
            }
    except Exception as e:
        checks["memory"] = {"status": "warning", "message": f"Memory check unavailable: {e}"}

    # 3. Provider configuration readiness
    try:
        settings = get_settings()
        prov_name = settings.default_llm_provider
        if prov_name == "gemini":
            has_key = bool(settings.google_api_key and settings.google_api_key.get_secret_value())
            if not has_key:
                is_degraded = True
                checks["provider"] = {
                    "status": "warning",
                    "message": "Gemini selected as default provider but GOOGLE_API_KEY is not configured.",
                }
            else:
                checks["provider"] = {"status": "ok", "message": "Gemini provider is configured."}
        else:
            checks["provider"] = {
                "status": "ok",
                "message": f"Default provider '{prov_name}' is configured for endpoint {settings.ollama_host}.",
            }
    except Exception as e:
        is_degraded = True
        checks["provider"] = {"status": "warning", "message": f"Provider check failed: {e}"}

    if is_fatal:
        status_code = 503
        overall_status = "unhealthy"
    elif is_degraded:
        status_code = 200
        overall_status = "degraded"
    else:
        status_code = 200
        overall_status = "ready"

    payload = {
        "status": overall_status,
        "service": "devops-ai",
        "version": __version__,
        "checks": checks,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return status_code, payload
