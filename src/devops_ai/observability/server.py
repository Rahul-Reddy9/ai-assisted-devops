# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Lightweight HTTP server exposing Prometheus metrics and health endpoints."""

import http.server
import json
from socketserver import ThreadingMixIn
import threading
from typing import Optional
from prometheus_client import CONTENT_TYPE_LATEST
from devops_ai.core.logging import get_logger
from devops_ai.observability.health import evaluate_liveness, evaluate_readiness
from devops_ai.observability.metrics import get_prometheus_exposition

logger = get_logger("observability.server")


class _ObservabilityHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Handles HTTP requests for /metrics, /healthz, and /readyz."""

    def log_message(self, format: str, *args) -> None:
        """Suppress default stderr HTTP request logging."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if not path:
            path = "/"

        if path in ("/healthz", "/livez"):
            payload = evaluate_liveness()
            self._send_json(200, payload)
        elif path == "/readyz":
            status_code, payload = evaluate_readiness()
            self._send_json(status_code, payload)
        elif path == "/metrics":
            data = get_prometheus_exposition()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_json(
                404, {"error": "Not Found", "supported_paths": ["/healthz", "/readyz", "/metrics"]}
            )

    def _send_json(self, status_code: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MetricsServer:
    """Manages the lifecycle of the background observability HTTP server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9090) -> None:
        self.host = host
        self.port = port
        self._server: Optional[_ThreadedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the HTTP server on a background daemon thread."""
        if self._server is not None:
            return

        self._server = _ThreadedHTTPServer((self.host, self.port), _ObservabilityHTTPRequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            f"Observability server started at http://{self.host}:{self.port} (/metrics, /healthz, /readyz)"
        )

    def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            if self._thread is not None:
                self._thread.join(timeout=2.0)
                self._thread = None
            logger.info("Observability server stopped.")
