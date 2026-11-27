# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Tests for Docker Compose monitoring orchestration and Prometheus/Grafana provisioning."""

import json
from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
PROMETHEUS_CONFIG = REPO_ROOT / "prometheus" / "prometheus.yml"
GRAFANA_DATASOURCE = REPO_ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yaml"
GRAFANA_DASHBOARDS_PROV = REPO_ROOT / "grafana" / "provisioning" / "dashboards" / "dashboards.yaml"
GRAFANA_DASHBOARD_JSON = REPO_ROOT / "grafana" / "dashboards" / "devops_ai_overview.json"


@pytest.fixture(scope="module")
def compose_data() -> dict:
    """Parse and return root docker-compose.yml data."""
    assert COMPOSE_FILE.is_file(), f"docker-compose.yml not found at {COMPOSE_FILE}"
    raw = COMPOSE_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "docker-compose.yml must parse to a dictionary"
    return data


@pytest.fixture(scope="module")
def prometheus_data() -> dict:
    """Parse and return prometheus/prometheus.yml data."""
    assert PROMETHEUS_CONFIG.is_file(), f"prometheus.yml not found at {PROMETHEUS_CONFIG}"
    raw = PROMETHEUS_CONFIG.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "prometheus.yml must parse to a dictionary"
    return data


def test_compose_file_structure(compose_data: dict):
    """Verify compose file contains required top-level services, networks, and volumes."""
    assert "services" in compose_data, "Compose file must define 'services'"
    services = compose_data["services"]
    assert "devops-ai" in services, "Service 'devops-ai' must be defined"
    assert "prometheus" in services, "Service 'prometheus' must be defined"
    assert "grafana" in services, "Service 'grafana' must be defined"

    assert "networks" in compose_data, "Compose file must define 'networks'"
    assert "volumes" in compose_data, "Compose file must define 'volumes'"
    volumes = compose_data["volumes"]
    assert "prometheus_data" in volumes, "Volume 'prometheus_data' must be defined"
    assert "grafana_data" in volumes, "Volume 'grafana_data' must be defined"


def test_devops_ai_service_configuration(compose_data: dict):
    """Verify devops-ai service build, command, ports, and healthcheck."""
    svc = compose_data["services"]["devops-ai"]

    # Build context
    assert "build" in svc, "devops-ai service must configure 'build'"
    build_cfg = svc["build"]
    assert build_cfg.get("context") == ".", "Build context must be root '.'"
    assert build_cfg.get("dockerfile") == "Dockerfile", "Dockerfile must be root 'Dockerfile'"

    # Command executes metrics serve
    cmd = svc.get("command", [])
    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = str(cmd)
    assert "metrics" in cmd_str, "Command must invoke 'metrics'"
    assert "serve" in cmd_str, "Command must invoke 'serve'"
    assert "9090" in cmd_str, "Metrics server command must bind to port 9090"

    # Port mapping
    ports = [str(p) for p in svc.get("ports", [])]
    assert any("9090:9090" in p or "9090" in p for p in ports), "Must expose port 9090"

    # Health check
    assert "healthcheck" in svc, "devops-ai service must specify healthcheck"
    hc = svc["healthcheck"]
    test_cmd = hc.get("test", [])
    test_str = " ".join(test_cmd) if isinstance(test_cmd, list) else str(test_cmd)
    assert "devops-ai" in test_str and "health" in test_str and "check" in test_str


def test_prometheus_service_configuration(compose_data: dict):
    """Verify prometheus service image, port mapping, and volume mounts."""
    svc = compose_data["services"]["prometheus"]

    # Image
    image = svc.get("image", "")
    assert "prom/prometheus" in image, f"Prometheus must use official image, got: {image}"

    # Port mapping: host 9091 -> container 9090 to avoid conflict with devops-ai 9090
    ports = [str(p) for p in svc.get("ports", [])]
    assert any("9091:9090" in p for p in ports), (
        "Prometheus must map host port 9091 to container 9090 to avoid host collisions"
    )

    # Volumes
    volumes = [str(v) for v in svc.get("volumes", [])]
    assert any("prometheus.yml:/etc/prometheus/prometheus.yml:ro" in v for v in volumes), (
        "Must mount prometheus.yml read-only"
    )
    assert any("prometheus_data:/prometheus" in v for v in volumes), (
        "Must mount persistent prometheus_data volume"
    )

    # Dependency on devops-ai health
    depends = svc.get("depends_on", {})
    if isinstance(depends, dict):
        assert "devops-ai" in depends, "Prometheus must depend on devops-ai"
        assert depends["devops-ai"].get("condition") == "service_healthy"
    elif isinstance(depends, list):
        assert "devops-ai" in depends


def test_grafana_service_configuration(compose_data: dict):
    """Verify grafana service image, port mapping, provisioning volumes, and credentials."""
    svc = compose_data["services"]["grafana"]

    # Image
    image = svc.get("image", "")
    assert "grafana/grafana" in image, f"Grafana must use official image, got: {image}"

    # Port mapping: 3000:3000
    ports = [str(p) for p in svc.get("ports", [])]
    assert any("3000:3000" in p or "3000" in p for p in ports), "Must map port 3000"

    # Configurable credentials via environment variables
    env = svc.get("environment", {})
    env_str = str(env)
    assert "GF_SECURITY_ADMIN_USER" in env_str, "Must configure admin user via environment"
    assert "GF_SECURITY_ADMIN_PASSWORD" in env_str, "Must configure admin password via environment"
    # Ensure no hardcoded raw secret passwords (e.g. must allow override or use variable)
    assert "${GRAFANA_ADMIN_PASSWORD" in env_str or "GF_SECURITY_ADMIN_PASSWORD" in env_str

    # Volumes
    volumes = [str(v) for v in svc.get("volumes", [])]
    assert any("provisioning/datasources" in v for v in volumes), (
        "Must mount datasources provisioning"
    )
    assert any("provisioning/dashboards" in v for v in volumes), (
        "Must mount dashboards provisioning"
    )
    assert any("dashboards" in v for v in volumes), "Must mount dashboard definitions"
    assert any("grafana_data:/var/lib/grafana" in v for v in volumes), (
        "Must mount persistent grafana_data volume"
    )


def test_prometheus_scrape_configuration(prometheus_data: dict):
    """Verify Prometheus scrape configuration targets devops-ai:9090 on /metrics."""
    assert "scrape_configs" in prometheus_data, "Prometheus config must define scrape_configs"
    jobs = {j.get("job_name"): j for j in prometheus_data["scrape_configs"]}

    assert "devops-ai" in jobs, "Prometheus must include scrape job 'devops-ai'"
    job = jobs["devops-ai"]
    assert job.get("metrics_path") == "/metrics", "Metrics path must be '/metrics'"

    # Verify target uses service DNS name
    static_cfgs = job.get("static_configs", [])
    assert len(static_cfgs) > 0, "Job must specify static_configs"
    targets = static_cfgs[0].get("targets", [])
    assert "devops-ai:9090" in targets, (
        f"Prometheus must scrape 'devops-ai:9090' using Docker DNS, got: {targets}"
    )


def test_grafana_datasource_provisioning():
    """Verify Grafana datasource YAML correctly points to Prometheus service."""
    assert GRAFANA_DATASOURCE.is_file(), f"Datasource file missing at {GRAFANA_DATASOURCE}"
    raw = GRAFANA_DATASOURCE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict)

    datasources = data.get("datasources", [])
    assert len(datasources) > 0, "Must define at least one datasource"
    ds = datasources[0]
    assert ds.get("type") == "prometheus", "Datasource type must be prometheus"
    assert ds.get("url") == "http://prometheus:9090", (
        f"Datasource url must point to 'http://prometheus:9090', got: {ds.get('url')}"
    )


def test_grafana_dashboard_provisioning_and_json():
    """Verify Grafana dashboard provider and overview dashboard JSON validity."""
    assert GRAFANA_DASHBOARDS_PROV.is_file(), (
        f"Dashboards provider file missing at {GRAFANA_DASHBOARDS_PROV}"
    )
    prov_data = yaml.safe_load(GRAFANA_DASHBOARDS_PROV.read_text(encoding="utf-8"))
    providers = prov_data.get("providers", [])
    assert len(providers) > 0, "Must define at least one dashboard provider"
    provider_path = providers[0].get("options", {}).get("path", "")
    assert provider_path in (
        "/etc/grafana/dashboards",
        "/var/lib/grafana/dashboards",
        "/etc/grafana/provisioning/dashboards",
    ), f"Unexpected dashboard provider path: {provider_path}"

    # Verify devops_ai_overview.json
    assert GRAFANA_DASHBOARD_JSON.is_file(), f"Dashboard JSON missing at {GRAFANA_DASHBOARD_JSON}"
    dashboard = json.loads(GRAFANA_DASHBOARD_JSON.read_text(encoding="utf-8"))
    assert "DevOps AI" in dashboard.get("title", ""), "Dashboard title must reference DevOps AI"
    assert len(dashboard.get("panels", [])) >= 5, "Dashboard must contain visualization panels"
