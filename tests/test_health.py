# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for application health endpoints and HTTP server."""

import json
import time
from unittest.mock import patch
import urllib.request
from devops_ai.observability.health import evaluate_liveness, evaluate_readiness
from devops_ai.observability.server import MetricsServer


def test_evaluate_liveness():
    live = evaluate_liveness()
    assert live["status"] == "healthy"
    assert live["service"] == "devops-ai"
    assert "version" in live
    assert live["uptime_seconds"] >= 0.0
    assert "timestamp" in live


def test_evaluate_readiness_healthy():
    status_code, ready = evaluate_readiness()
    assert status_code == 200
    assert ready["status"] in ("ready", "degraded")
    assert "checks" in ready
    assert "filesystem" in ready["checks"]
    assert ready["checks"]["filesystem"]["status"] == "ok"


def test_evaluate_readiness_filesystem_failure():
    with patch("tempfile.NamedTemporaryFile", side_effect=OSError("Read-only file system")):
        status_code, ready = evaluate_readiness()
        assert status_code == 503
        assert ready["status"] == "unhealthy"
        assert ready["checks"]["filesystem"]["status"] == "failed"


def test_metrics_http_server_endpoints():
    server = MetricsServer(host="127.0.0.1", port=0)
    server.start()
    port = server.bound_port

    try:
        # Allow threaded server socket event loop to initialize
        time.sleep(0.05)

        # 1. Test /healthz
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"

        # 2. Test /readyz
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] in ("ready", "degraded")

        # 3. Test /metrics
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3.0) as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "devops_ai_" in content

        # 4. Test 404
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nonexistent", timeout=3.0)
            assert False, "Expected 404 HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()
