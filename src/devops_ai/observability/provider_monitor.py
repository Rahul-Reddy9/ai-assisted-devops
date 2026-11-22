# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Telemetry prober and health monitor for Ollama and Gemini LLM providers."""

from dataclasses import dataclass
import time
from typing import Dict, Optional
from devops_ai.core.logging import get_logger
from devops_ai.llm.router import LLMRouter
from devops_ai.observability.metrics import record_provider_health

logger = get_logger("observability.provider_monitor")


@dataclass
class ProviderTelemetry:
    """Telemetry report for an individual LLM provider."""

    provider: str
    is_healthy: bool
    probe_latency_seconds: float
    model: str
    endpoint: Optional[str] = None
    error_message: Optional[str] = None


class ProviderMonitor:
    """Probes LLM providers, records latency, and updates Prometheus telemetry."""

    def probe_provider(
        self,
        provider_name: str,
        timeout: Optional[float] = None,
    ) -> ProviderTelemetry:
        """Probe single provider connectivity, update metrics, and return telemetry."""
        safe_name = provider_name.lower().strip()
        start = time.monotonic()
        is_healthy = False
        err_msg: Optional[str] = None
        model_name = ""
        host_endpoint: Optional[str] = None

        try:
            provider = LLMRouter.get_provider(provider_name=safe_name, timeout=timeout)
            model_name = provider.model_name
            host_endpoint = provider.host

            is_healthy = provider.check_health(verify_remote=True)
            if not is_healthy:
                err_msg = f"Provider '{safe_name}' remote verification check failed."
        except Exception as e:
            err_msg = str(e)
            logger.debug(f"Provider '{safe_name}' probe failed: {e}")

        latency = max(0.0, time.monotonic() - start)
        record_provider_health(safe_name, is_healthy, latency)

        return ProviderTelemetry(
            provider=safe_name,
            is_healthy=is_healthy,
            probe_latency_seconds=round(latency, 4),
            model=model_name,
            endpoint=host_endpoint,
            error_message=err_msg,
        )

    def probe_all(self, timeout: Optional[float] = None) -> Dict[str, ProviderTelemetry]:
        """Probe all supported LLM providers and record telemetry."""
        results: Dict[str, ProviderTelemetry] = {}
        for prov in ("ollama", "gemini"):
            results[prov] = self.probe_provider(prov, timeout=timeout)
        return results
