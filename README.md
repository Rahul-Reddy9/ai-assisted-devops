# DevOps AI

DevOps AI is an enterprise-grade AI-assisted DevOps engineering suite designed to automate infrastructure generation, system observability, log analysis, and incident remediation.

## Architecture Overview
- **Pluggable LLM Providers**: Unified interface supporting local execution via Ollama (default) and hosted execution via Google Gemini.
- **Infrastructure Generators**: Best-practice Dockerfile and multi-document Kubernetes manifest generation with strict structural validation and static heuristic analysis.
- **Safe File Writing**: Atomic file writing utility with temporary file staging and overwrite protection.
- **Unified CLI**: Developer-friendly CLI built on Typer and Rich with error handling, markup escaping, and best-effort credential redaction.

## Installation

### Core Installation
```bash
pip install -e .
```

### Provider Extras
```bash
# For local Ollama support
pip install -e ".[ollama]"

# For Google Gemini support
pip install -e ".[gemini]"

# For development, full providers, and future Phase dependencies
pip install -e ".[full,dev]"
```

## Configuration
Copy the example environment file:
```bash
cp .env.example .env
```
Key configuration settings:
- `DEFAULT_LLM_PROVIDER`: Set to `ollama` (default) or `gemini`.
- `OLLAMA_MODEL`: Default `llama3.2:1b`.
- `OLLAMA_HOST`: Default `http://localhost:11434`.
- `GOOGLE_API_KEY`: API key for Gemini (stored securely as a SecretStr).
- `GEMINI_MODEL`: Default `gemini-2.5-flash`.
- `LLM_TIMEOUT_SECONDS`: Default `60.0`.
- `ENABLE_LLM_FALLBACK`: Set to `true` to enable fallback to alternate provider on failure (default `false`).

## CLI Usage
```bash
# Check provider health (local configuration check)
devops-ai provider check --provider ollama
devops-ai provider check --provider gemini

# Check provider health with live remote verification
devops-ai provider check --provider ollama --verify-remote
devops-ai provider check --provider gemini --verify-remote

# Generate Dockerfile
devops-ai generate dockerfile python --port 8000 --output Dockerfile

# Generate Kubernetes manifests
devops-ai generate k8s myapp --image myapp:1.0 --port 8080 --service-type ClusterIP --output deployment.yaml
```

## Security & Validation Notes
- **Static Heuristic Validation:** Built-in Dockerfile and Kubernetes validators perform static structural and syntax checks. These checks catch common configuration errors but do not replace container image vulnerability scanners (e.g., Trivy) or cluster admission controllers (e.g., OPA Gatekeeper, Kyverno).
- **Secret Protection:** Best-effort redaction of configured Google API keys and recognized key patterns is applied to known error-display paths. This is not a universal secret-detection guarantee. API keys are loaded strictly from the environment or `.env` into masked `SecretStr` containers.
