# DevOps AI

DevOps AI is an enterprise-grade AI-assisted DevOps engineering suite designed to automate infrastructure generation, system observability, log analysis, and incident remediation.

## Architecture Overview
- **Pluggable LLM Providers**: Unified interface supporting local execution via Ollama (default) and hosted execution via Google Gemini with automated fallback support.
- **Infrastructure Generators**: Best-practice Dockerfile and multi-document Kubernetes manifest generation with strict structural validation, port checks, and static heuristic analysis.
- **Safe File Writing**: Atomic file writing utility with temporary file staging, fsync synchronization, and overwrite protection.
- **Observability & Telemetry**: Low-cardinality Prometheus metrics registry, thread-safe caching, multi-threaded metrics and health probe server (`/metrics`, `/healthz`, `/readyz`).
- **Structured Logging & Secret Scrubbing**: Standardized JSON formatting, ISO-8601 timestamps, and regex-based credential scrubbing across logs, terminal displays, and errors.
- **Read-Only Infrastructure Monitoring**: Graceful, non-mutating inspection of Docker containers and Kubernetes workloads (Deployments, Pods, Services) with offline resilience.
- **Log Anomaly Detection**: Pure-Python sliding-window temporal burst detection and machine-learning Isolation Forest anomaly identification.
- **Autonomous AI Agent Architecture**: Extensible agent lifecycle (`BaseAgent`), multi-agent coordination (`AgentOrchestrator`), sequential pipelining with fact accumulation, broadcast routing, strict read-only safety boundary, and zero automated execution hooks.
- **RAG Document Ingestion & Chunking**: Modular ingestion pipeline for Markdown, Plain Text, PDF, JSON, and YAML with structure-aware chunking, code block preservation, secret scrubbing, and deduplication.
- **Turnkey Grafana Integration**: Pre-provisioned Grafana dashboards and Prometheus datasource configurations.
- **Unified CLI**: Developer-friendly CLI built on Typer and Rich with error handling, markup escaping, and comprehensive credential redaction.

---

## Installation

### Core Installation
```bash
pip install -e .
```

### Optional Extras
```bash
# For local Ollama support
pip install -e ".[ollama]"

# For Google Gemini support
pip install -e ".[gemini]"

# For observability (Prometheus client, psutil, Kubernetes client)
pip install -e ".[observability]"

# Full installation (all providers, ML anomaly detection, observability, dev tooling)
pip install -e ".[full,dev]"
```

---

## Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Key configuration settings:
- `DEFAULT_LLM_PROVIDER`: Set to `ollama` (default) or `gemini`.
- `OLLAMA_MODEL`: Default `llama3.2:1b`.
- `OLLAMA_HOST`: Default `http://localhost:11434`.
- `GOOGLE_API_KEY`: API key for Gemini (stored securely in a masked `SecretStr`).
- `GEMINI_MODEL`: Default `gemini-2.5-flash`.
- `LLM_TIMEOUT_SECONDS`: Default `60.0`.
- `ENABLE_LLM_FALLBACK`: Set to `true` to enable fallback to alternate provider on failure (default `false`).
- `LOG_FORMAT`: Format for application logs: `json` or `text` (default `text`).
- `METRICS_HOST`: Host interface to bind Prometheus metrics server (default `127.0.0.1`).
- `METRICS_PORT`: Port to bind Prometheus metrics server (default `9090`).
- `HEALTH_CHECK_TIMEOUT_SECONDS`: Maximum timeout for provider health telemetry probes (default `3.0`).

---

## CLI Usage

### 1. LLM Provider Checks & Infrastructure Generation
```bash
# Check provider health (local configuration check)
devops-ai provider check --provider ollama
devops-ai provider check --provider gemini

# Check provider health with live remote verification
devops-ai provider check --provider ollama --verify-remote
devops-ai provider check --provider gemini --verify-remote

# Generate Dockerfile with non-root security validation
devops-ai generate dockerfile python --port 8000 --output Dockerfile

# Generate Kubernetes manifests (Deployment and Service)
devops-ai generate k8s myapp --image myapp:1.0 --port 8080 --service-type ClusterIP --output deployment.yaml
```

### 2. Health & Telemetry Probes
```bash
# Evaluate process liveness and readiness
devops-ai health check

# Evaluate health and output structured JSON (ideal for CI/CD or k8s probes)
devops-ai health check --json

# Start the Prometheus metrics and health HTTP server
devops-ai metrics serve --host 127.0.0.1 --port 9090

# Dump current Prometheus metrics snapshot directly to stdout
devops-ai metrics dump
```

### 3. Read-Only Infrastructure Monitoring
```bash
# Safely inspect Docker containers (read-only)
devops-ai monitor docker
devops-ai monitor docker --running
devops-ai monitor docker --json

# Safely inspect Kubernetes Deployments, Pods, and Services (read-only)
devops-ai monitor k8s
devops-ai monitor k8s --namespace production
devops-ai monitor k8s --json
```

### 4. Log Analytics & Anomaly Detection
```bash
# Detect error bursts with a 30s sliding window and threshold of 3 errors
devops-ai log-analyze /path/to/app.log --window 30 --threshold 3

# Detect volume anomalies using Scikit-Learn IsolationForest
devops-ai log-analyze /path/to/app.log --ml

# Output analysis results as JSON
devops-ai log-analyze /path/to/app.log --json
```

### 5. Incident Lifecycle Management & Two-Tier RCA
```bash
# List all recorded operational incidents (filtered by status, severity, or activity)
devops-ai incident list
devops-ai incident list --status DETECTED --severity CRITICAL
devops-ai incident list --active-only
devops-ai incident list --json

# Inspect comprehensive details, correlated evidence, and timeline of an incident
devops-ai incident show INC-20261124-0001
devops-ai incident show INC-20261124-0001 --json

# Perform two-tier root cause analysis (Tier 1 deterministic + conditional Tier 2 AI)
devops-ai incident analyze INC-20261124-0001
devops-ai incident analyze INC-20261124-0001 --force-tier2
devops-ai incident analyze INC-20261124-0001 --skip-tier2
devops-ai incident analyze INC-20261124-0001 --json

# Run targeted RCA diagnosis and view risk-categorized remediation advisory
devops-ai incident rca INC-20261124-0001
devops-ai incident rca INC-20261124-0001 --force-tier2
devops-ai incident rca INC-20261124-0001 --json

# Transition incident through resolution and closure with operator notes
devops-ai incident resolve INC-20261124-0001 --notes "Increased pod memory limit to 1Gi"
devops-ai incident close INC-20261124-0001 --notes "Post-mortem completed and reviewed"

# Export structured incident post-mortem reports (Markdown, JSON, or Terminal)
devops-ai incident report INC-20261124-0001 --format markdown --output postmortem.md
devops-ai incident report INC-20261124-0001 --format json --output incident.json
devops-ai incident report INC-20261124-0001 --format terminal
```

---

## Prometheus Metrics Reference

| Metric Name | Type | Description | Labels |
|---|---|---|---|
| `devops_ai_generation_requests_total` | Counter | Total infrastructure manifest generation attempts | `generator`, `provider`, `status` |
| `devops_ai_generation_duration_seconds` | Histogram | Latency distribution of manifest generations | `generator`, `provider` |
| `devops_ai_validation_warnings_total` | Counter | Total static validation warnings emitted | `generator`, `warning_type` |
| `devops_ai_validation_errors_total` | Counter | Total fatal validation errors | `generator`, `error_code` |
| `devops_ai_provider_health_status` | Gauge | LLM provider health status (1 = healthy, 0 = degraded) | `provider` |
| `devops_ai_provider_probe_duration_seconds` | Gauge | Round-trip duration of recent health probes | `provider` |
| `devops_ai_docker_available` | Gauge | Local Docker daemon connectivity (1 = available, 0 = offline) | None |
| `devops_ai_k8s_available` | Gauge | Kubernetes cluster connectivity (1 = available, 0 = offline) | None |
| `devops_ai_log_anomalies_detected_total` | Counter | Operational anomalies and error bursts detected | `detection_type` |
| `devops_ai_system_cpu_percent` | Gauge | Host CPU utilization percentage | None |
| `devops_ai_system_memory_bytes` | Gauge | Host memory in use | None |

---

## Grafana & Dashboards

A turnkey dashboard and datasource configuration is located in the `grafana/` directory.

To launch Prometheus and Grafana locally with automated dashboard provisioning, see [grafana/README.md](grafana/README.md).

---

---

## Incident Analysis & Two-Tier Investigation

DevOps AI provides an autonomous two-tier incident investigation and remediation platform designed for production site reliability engineering.

### Two-Tier Architecture & Orchestration
1. **Tier 1 — Deterministic Analysis (100% Offline):**
   - Ingests alerts across logs, metrics, container states, health probes, and Kubernetes events.
   - Evaluates deterministic failure patterns (Confirmed OOMKilled vs Ambiguous SIGKILL, CrashLoopBackOff, ImagePullBackOff, Database Connection Refused, CPU Throttling, Memory Saturation, Readiness Probe Failures, and LLM Rate Limiting).
   - Calculates bounded mathematical evidence confidence:
     $$\text{Confidence} = \text{clamp}\left((S_{\text{base}} \times W_{\text{source}}) + S_{\text{corrob}} - P_{\text{contradict}}, 0.0, 1.0\right)$$
   - Executes in milliseconds entirely offline without LLM dependencies.

2. **Tier 2 — AI-Assisted Semantic Investigation:**
   - Evaluates deep semantic context using local Ollama models or hosted Google Gemini with automated fallback.
   - Automatically escalates when deterministic criteria require deep investigation:
     - Top hypothesis confidence below threshold (`< 0.85`).
     - Application or unknown category incidents.
     - Ambiguous competing hypotheses with delta $< 0.15$.
     - Top hypothesis is unsupported by catalog citations.
   - Preserves deterministic Tier 1 findings if Tier 2 fails, times out, or produces ungrounded output.

### Evidence Grounding & Confidence Provenance
- **Evidence Verification:** Every hypothesis must cite verifiable evidence references (`EVID-*`) present in the incident evidence catalog.
- **Hallucination Rejection:** Unknown evidence IDs are automatically stripped and penalized.
- **Unsupported Hypothesis Capping:** Hypotheses lacking verified supporting evidence citations are strictly marked `is_unsupported = True` and confidence is capped at $\le 0.30$.
- **No LLM Confidence Inflation:** LLM self-reported confidence is never treated as calibrated probability. Final confidence is mathematically derived from verified evidence citations and corroborating signals.

### Remediation Advisory & Safety Guard Rails
- **Advisory Only:** Remediation actions are strictly advisory recommendations for human engineers. **Automated execution is completely disabled**; zero execution hooks exist in the codebase.
- **Risk Classification:**
  - `SAFE`: Read-only diagnostic and inspection commands (e.g. `kubectl logs`, `docker inspect`, `kubectl get events`).
  - `MODERATE`: Workload-affecting mutating operations (e.g. `kubectl rollout restart`, `docker restart`, resource updates).
  - `DANGEROUS`: Destructive, irreversible, or high-impact actions (e.g. `kubectl delete`, `rm -rf`, `DROP TABLE`, `kill -9`).
- **Automated Guard Rails:** Destructive commands are scanned by deterministic safety classifiers and are **never** classified as `SAFE` or `MODERATE`. Every action includes human verification steps, preconditions, and rollback guidance.

### Security & Prompt Injection Defense
- All incident telemetry, logs, container output, and event messages are treated as **untrusted input**.
- Evidence is isolated inside `<UNTRUSTED_INCIDENT_EVIDENCE>` XML delimiters in prompts.
- Models are strictly instructed to treat evidence as passive data and ignore any embedded directives or instruction overrides.
- Automated secret scrubbing sanitizes API keys, passwords, and bearer tokens across prompts, outputs, logs, and diagnostic metadata.

---

## AI Agent Architecture & Multi-Agent Orchestration (Phase 5)

DevOps AI provides an extensible, modular AI agent framework designed for production site reliability engineering diagnostics, multi-step investigation, and multi-agent coordination.

### Core Architecture & Lifecycle
All agent implementations inherit from `BaseAgent` and participate in an explicit state machine:
$$\text{IDLE} \longrightarrow \text{PLANNING} \longrightarrow \text{EXECUTING} \longrightarrow \text{COMPLETED} \ / \ \text{FAILED}$$

- **`BaseAgent`**: Abstract foundation managing lifecycle transitions, execution limits, untrusted context boundary sanitation, and integration with `LLMRouter`.
- **`AgentLimits`**: Configurable per-agent constraints:
  - `read_only`: Strict boolean defaulting to `True`. Instantiating an agent with `read_only=False` raises `AgentSecurityError`.
  - `max_iterations`: Bounded reasoning/action loops (default: 10, range: 1–50).
  - `timeout_seconds`: Wall-clock execution timeout in seconds (default: 60.0s).
  - `max_input_chars`: Untrusted input character boundary (default: 16,000 characters).
- **`AgentTask`**: Immutable assigned goal with task-specific context data, read-only validation, and optional timeout overrides.
- **`AgentStep`**: Discrete reasoning or observation step tracking thought process, action, and diagnostic findings.
- **`AgentResult`**: Structured output encompassing task status, concise executive summary, structured findings, complete step history, and execution metrics.
- **`AgentContext`**: Shared context container maintaining step history, accumulated facts across agents, runtime variables, and scratchpad storage.

### Orchestration Layer (`AgentOrchestrator`)
The `AgentOrchestrator` coordinates agent registries, lifecycle resets, and task routing patterns:
1. **Single Agent Dispatch (`dispatch`)**: Routes an `AgentTask` directly to a registered agent by name or falls back to the default registered agent.
2. **Sequential Multi-Agent Pipelines (`run_pipeline`)**: Executes a sequence of agents where upstream outputs are chained into downstream context:
   - Upstream findings are saved to `context.variables["<agent_name>_findings"]`.
   - Extracted diagnostic facts are accumulated directly into `context.facts`.
   - The pipeline aborts immediately if any agent fails, returning partial execution history.
3. **Broadcast Execution (`broadcast`)**: Dispatches the same task in parallel or sequence across all registered agents and aggregates results.
4. **Capability Search (`get_agents_by_capability`)**: Locates registered agents matching specific capabilities (e.g. `ANALYZE_LOGS`, `INCIDENT_DIAGNOSIS`).

### LLM Router Integration & Fallback
- Agents leverage `LLMRouter.generate_with_fallback(...)` directly for semantic reasoning and completion generation.
- Respects provider timeouts, temperature overrides, and automatic failover between Ollama and Google Gemini.
- Does not create redundant LLM client layers; standardizes on existing project provider infrastructure.
- Provides graceful deterministic fallback if all configured LLM providers fail or are unreachable.

### Strict Read-Only Safety Boundary & Zero-Execution Model
- **Strict Read-Only by Default**: All agents operate in read-only diagnostic mode. Any attempt to initialize an agent with `read_only=False` or pass a mutating task raises `AgentSecurityError`.
- **Prohibited Mutation Blocking**: Tasks or actions attempting destructive mutations (such as `rm -rf`, `kubectl delete`, `docker rm`, `DROP TABLE`, `kill -9`) are deterministically blocked prior to execution.
- **Untrusted Context Boundary & Delimiter Neutralization**: All telemetry, logs, and external payloads are length-bounded and neutralized to prevent prompt injection or escape attempts.
- **Automated Secret Scrubbing**: Passwords, API tokens, bearer keys, and sensitive environment variables are sanitized from agent prompts, step history, and output findings.
- **Zero Execution Hooks**: Agent outputs and remediation recommendations are strictly advisory for human site reliability engineers. No automated command execution hooks exist.

### Reference DevOps Diagnostic Agent (`DevOpsDiagnosticAgent`)
A reference implementation is included out-of-the-box:
- Evaluates raw diagnostic context, container exit codes (e.g. exit code 137 / OOMKilled), and log burst alerts.
- Produces structured findings with identified anomalies, severity assessments, and advisory remediation steps.
- Operates reliably both with live LLMs and offline via rule-based deterministic fallback.

---

## RAG Document Ingestion & Chunking Subsystem (Phase 5 Commit 2)

DevOps AI includes a modular document ingestion, normalization, and chunking subsystem designed to prepare operational knowledge, incident runbooks, architecture diagrams, and system configurations for semantic retrieval and vector indexing.

### Architecture & Processing Lifecycle
```
Raw Documents (.md, .txt, .pdf, .json, .yaml)
               │
               ▼
   [ Format Extractors ]  ──► Validates extensions, enforces file size limits, safe parsing
               │
               ▼
   [ Text Preprocessor ] ──► Normalizes CRLF, strips control bytes, scrubs secrets, SHA-256 hash
               │
               ▼
   [ Smart Chunker ]     ──► Preserves markdown code blocks, sentence/paragraph splits, overlap
               │
               ▼
    List[DocumentChunk]   ──► Deterministic IDs, indexed chunks, rich provenance metadata
```

### Supported Document Formats
| Format | Extensions | Extractor Details |
|---|---|---|
| **Markdown** | `.md`, `.markdown` | Extracts document title from top-level header (`# Title`), preserves tables, lists, and code blocks. |
| **Plain Text** | `.txt`, `.text` | UTF-8 / UTF-8-sig safe text extraction. |
| **PDF** | `.pdf` | Multi-page text extraction via `pypdf`, extracts metadata (`/Title`), rejects encrypted PDFs. |
| **JSON** | `.json` | Safe parser (`json.loads`), normalizes formatting with 2-space indentation, tracks key counts. |
| **YAML** | `.yaml`, `.yml` | Safe parser (`yaml.safe_load_all`), supports multi-document streams, tracks document counts. |

Executable and script extensions (`.exe`, `.sh`, `.py`, `.bin`, `.dll`, `.bat`, etc.) are strictly prohibited and rejected prior to processing.

### Intelligent Chunking Strategy (`SmartChunker`)
- **Structure-Aware Splitting**: Groups text into semantic paragraph and code block units. Fenced code blocks (` ```...``` `) are kept together as atomic blocks whenever they fit within `chunk_size`.
- **Configurable Size & Overlap**: Default `chunk_size=1000` characters with `chunk_overlap=150` characters.
- **Safe Oversized Partitioning**: Blocks larger than `chunk_size` are sub-split along line breaks (for code) or sentence boundaries (`. `, `? `, `! `) without data loss or truncation.
- **Deterministic Identifiers**: Generates stable chunk IDs formatted as `CHK-<doc_id>-<chunk_index:04d>` with synchronized `total_chunks` and content hashes.

### Security & Defensive Controls
- **Strict Read-Only Execution**: The subsystem operates in a strictly passive, read-only manner with no filesystem mutation outside transient processing.
- **Path Traversal Defense**: All input paths are resolved and validated. Traversals containing `..` or escaping an optional `allowed_root` directory raise `DocumentSecurityError`.
- **File Size & Input Limits**: Enforces configurable `rag_max_file_size_bytes` (default: 10MB) to prevent memory exhaustion or zip-bomb style denial of service.
- **Passive Untrusted Context**: Document content is treated purely as untrusted data. Embedded instructions (e.g. `<SYSTEM_MANDATE>`) or commands (`rm -rf`, `kubectl delete`) are never executed.
- **Automated Secret Scrubbing**: Credentials, API tokens, passwords, and bearer keys within operational documents are scrubbed prior to chunk generation.

### Example Usage
```python
from pathlib import Path
from devops_ai.rag import DocumentIngestionPipeline, DocumentFormat

# Initialize pipeline with custom chunk parameters
pipeline = DocumentIngestionPipeline(chunk_size=800, chunk_overlap=100)

# Ingest an incident runbook
result = pipeline.ingest_file(
    file_path=Path("docs/runbooks/database_failover.md"),
    allowed_root=Path("docs"),
    custom_metadata={"service": "payment-db", "tier": "critical"},
)

print(f"Ingested {result.total_chunks} chunks for doc: {result.document_id}")
for chunk in result.chunks:
    print(f"Chunk {chunk.chunk_index}: {len(chunk.content)} chars (has_code={chunk.metadata['has_code_block']})")
```

---

## Security & Reliability Principles
- **Read-Only Operations:** All container and Kubernetes monitoring commands operate in a strictly non-mutating manner. The suite never modifies, scales, restarts, or deletes running workloads.
- **Zero-Hang Timeout Enforcement:** All network calls, health probes, Docker queries, and Kubernetes API requests enforce strict timeouts, ensuring commands never hang when external services or daemons are unreachable.
- **Secret Scrubbing:** Credentials, API tokens, passwords, and authorization headers are sanitized and redacted from logs, CLI output, and error messages via regex pattern matching and `SecretStr` masking.
- **Static Heuristic Validation:** Built-in validators catch common security vulnerabilities (e.g. root containers, exposed internal ports, invalid Kubernetes replica counts) prior to deployment.
