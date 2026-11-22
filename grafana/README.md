# DevOps AI — Grafana & Prometheus Observability Stack

This directory contains turnkey Prometheus datasources and Grafana dashboard provisioning configurations for visualizing `devops-ai` operational metrics.

---

## 1. Directory Structure

```text
grafana/
├── dashboards/
│   └── devops_ai_overview.json          # Production dashboard (v10+ compatible)
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yaml              # Auto-configured Prometheus datasource
│   └── dashboards/
│       └── dashboards.yaml              # Auto-provisioning provider definition
└── README.md                            # Documentation and quickstart guide
```

---

## 2. Quickstart with Docker Compose

To run Prometheus and Grafana locally with automated provisioning:

1. Create a `docker-compose.monitoring.yaml` file in your root workspace:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.52.0
    container_name: devops-ai-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:10.4.0
    container_name: devops-ai-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning/datasources:/etc/grafana/provisioning/datasources:ro
      - ./grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus
```

2. Create a `prometheus.yml` scrape configuration file:

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'devops-ai'
    static_configs:
      - targets: ['host.docker.internal:9090']
```

3. Start the metrics server in `devops-ai`:

```bash
devops-ai metrics serve --host 0.0.0.0 --port 9090
```

4. Launch the monitoring stack:

```bash
docker compose -f docker-compose.monitoring.yaml up -d
```

5. Open Grafana at [http://localhost:3000](http://localhost:3000) (Login: `admin` / `admin`).
   Navigate to **Dashboards > DevOps AI > DevOps AI — Observability & Telemetry**.

---

## 3. Metrics Reference

| Metric Name | Type | Description | Labels |
|---|---|---|---|
| `devops_ai_generation_requests_total` | Counter | Total manifest generation attempts | `generator`, `provider`, `status` |
| `devops_ai_generation_duration_seconds` | Histogram | Latency distribution of generations | `generator`, `provider` |
| `devops_ai_validation_warnings_total` | Counter | Total heuristic warnings | `generator`, `warning_type` |
| `devops_ai_validation_errors_total` | Counter | Total fatal validation errors | `generator`, `error_code` |
| `devops_ai_provider_health_status` | Gauge | Connectivity status (1=OK, 0=Fail) | `provider` |
| `devops_ai_provider_probe_duration_seconds` | Gauge | Round-trip latency of health probes | `provider` |
| `devops_ai_docker_available` | Gauge | Local Docker daemon status (1/0) | None |
| `devops_ai_k8s_available` | Gauge | Kubernetes cluster status (1/0) | None |
| `devops_ai_log_anomalies_detected_total` | Counter | Operational anomalies identified | `detection_type` |
| `devops_ai_system_cpu_percent` | Gauge | Host CPU utilization percentage | None |
| `devops_ai_system_memory_bytes` | Gauge | Host memory in use | None |
