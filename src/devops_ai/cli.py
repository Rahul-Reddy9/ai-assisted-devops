# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Command-line interface entry point for DevOps AI."""

from dataclasses import asdict
import datetime
import json
import logging
from pathlib import Path
import re
import time
from typing import Dict, Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
import typer

from devops_ai import __version__
from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import (
    BillingDataError,
    ConcurrencyConflictError,
    CostAnomalyError,
    CostForecastError,
    CurrencyMismatchError,
    DatasetValidationError,
    DevOpsAIError,
    FinOpsError,
    IncidentNotFoundError,
    InsufficientDataError,
    OptimizationError,
    RepositoryLockError,
)
from devops_ai.core.logging import setup_logging
from devops_ai.generators.dockerfile import DockerfileGenerator
from devops_ai.generators.k8s_manifest import K8sManifestGenerator
from devops_ai.incident import (
    AlertSource,
    Category,
    IncidentAnalyzer,
    IncidentReporter,
    IncidentRepository,
    IncidentStatus,
    Severity,
    TimelineEntry,
)
from devops_ai.llm.router import LLMRouter
from devops_ai.observability import (
    DockerMonitor,
    K8sMonitor,
    LogAnalyzer,
    LogParser,
    MetricsServer,
    evaluate_liveness,
    evaluate_readiness,
    get_prometheus_exposition,
    record_log_anomaly,
)
from devops_ai.finops import (
    AnomalySeverity,
    CloudProvider,
    CostAnalyzer,
    CostAnomalyDetector,
    CostForecaster,
    FinOpsAgent,
    ForecastMethod,
    OptimizationRisk,
    ResourceCategory,
    ResourceOptimizer,
    get_synthetic_anomaly_and_forecast_dataset,
    get_synthetic_billing_dataset,
    get_synthetic_resource_inventory,
    load_billing_records,
    load_cloud_resources,
    load_resource_utilization,
)
from devops_ai.observability.metrics import record_finops_operation
from devops_ai.rag.evaluation import (
    RAGEvaluator,
    get_builtin_evaluation_dataset,
    load_evaluation_dataset,
)

app = typer.Typer(
    name="devops-ai",
    help="Unified AI-Assisted DevOps suite for infrastructure generation, observability, and incident diagnosis.",
    add_completion=False,
)
generate_app = typer.Typer(
    name="generate", help="Generate infrastructure manifests (Dockerfiles, Kubernetes YAML)."
)
provider_app = typer.Typer(name="provider", help="Inspect and test LLM provider connectivity.")
health_app = typer.Typer(name="health", help="Evaluate process liveness and readiness.")
metrics_app = typer.Typer(name="metrics", help="Prometheus telemetry endpoints and inspection.")
monitor_app = typer.Typer(name="monitor", help="Inspect container and Kubernetes infrastructure.")
incident_app = typer.Typer(
    name="incident", help="Autonomous incident investigation, lifecycle management, and RCA."
)
rag_app = typer.Typer(
    name="rag", help="RAG document ingestion, semantic retrieval, and evaluation."
)
finops_app = typer.Typer(
    name="finops", help="Cloud and FinOps cost analysis, optimization, and forecasting."
)

app.add_typer(generate_app)
app.add_typer(provider_app)
app.add_typer(health_app)
app.add_typer(metrics_app)
app.add_typer(monitor_app)
app.add_typer(incident_app)
app.add_typer(rag_app)
app.add_typer(finops_app)

console = Console()


def _sanitize_cli_text(text: str) -> str:
    """Best-effort redaction of API keys before terminal display."""
    try:
        settings = get_settings()
        sanitized = text
        if settings.google_api_key:
            raw = settings.google_api_key.get_secret_value()
            if raw and raw in sanitized:
                sanitized = sanitized.replace(raw, "[REDACTED_API_KEY]")
        sanitized = re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", sanitized)
        return sanitized
    except Exception:
        return re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", text)


def version_callback(value: bool):
    if value:
        console.print(f"[bold green]devops-ai[/bold green] version [cyan]{__version__}[/cyan]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """DevOps AI command-line interface."""
    setup_logging()


@app.command()
def version():
    """Print the DevOps AI version."""
    console.print(f"devops-ai {__version__}")


@provider_app.command("check")
def check_provider(
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider name (ollama or gemini)."
    ),
    host: Optional[str] = typer.Option(None, "--host", help="Ollama host URL override."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override."),
    timeout: Optional[float] = typer.Option(None, "--timeout", "-t", help="Timeout in seconds."),
    verify_remote: bool = typer.Option(
        False, "--verify-remote", help="Perform live remote verification ping to LLM endpoint."
    ),
):
    """Check connectivity and health of the configured LLM provider."""
    try:
        if provider and provider.lower() not in ("ollama", "gemini"):
            console.print(
                Panel(
                    f"[bold red]Unsupported provider: '{escape(provider)}'. Must be 'ollama' or 'gemini'.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        if timeout is not None and timeout <= 0:
            console.print(
                Panel(
                    "[bold red]Timeout must be a positive number of seconds.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        llm = LLMRouter.get_provider(
            provider_name=provider, model=model, host=host, timeout=timeout
        )
        is_healthy = llm.check_health(verify_remote=verify_remote)

        if llm.provider_name == "gemini":
            if not is_healthy:
                console.print(
                    Panel(
                        "[bold red]Gemini check failed.[/bold red]\n"
                        "Ensure GOOGLE_API_KEY is configured and valid.",
                        title="Provider Status",
                    )
                )
                raise typer.Exit(code=1)
            elif not verify_remote:
                console.print(
                    Panel(
                        f"[bold green]Gemini credentials configured.[/bold green]\n"
                        f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                        f"[yellow]Local configuration verified; remote connectivity not tested.[/yellow]\n"
                        f"Pass [cyan]--verify-remote[/cyan] to test live API connectivity.",
                        title="Provider Status",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[bold green]Gemini is healthy and connected.[/bold green]\n"
                        f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                        f"Live remote verification succeeded.",
                        title="Provider Status",
                    )
                )
        else:
            endpoint = llm.host or "unknown"
            if is_healthy:
                if verify_remote:
                    from devops_ai.llm.ollama_provider import OllamaProvider

                    installed = []
                    if isinstance(llm, OllamaProvider):
                        installed = llm.get_installed_models()
                    model_status = (
                        f" (model '{escape(llm.model_name)}' installed)"
                        if llm.model_name in installed
                        else ""
                    )
                    console.print(
                        Panel(
                            f"[bold green]Provider 'ollama' is healthy and reachable.[/bold green]{model_status}\n"
                            f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                            f"Endpoint: [cyan]{escape(endpoint)}[/cyan]",
                            title="Provider Status",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            f"[bold green]Ollama configuration valid.[/bold green]\n"
                            f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                            f"Endpoint: [cyan]{escape(endpoint)}[/cyan]\n"
                            f"[yellow]Local configuration verified; pass --verify-remote to ping daemon.[/yellow]",
                            title="Provider Status",
                        )
                    )
            else:
                console.print(
                    Panel(
                        f"[bold red]Provider 'ollama' is unreachable at {escape(endpoint)}.[/bold red]\n"
                        f"Ensure Ollama daemon is running ('ollama serve').",
                        title="Provider Status",
                    )
                )
                raise typer.Exit(code=1)
    except DevOpsAIError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(
            Panel(f"[bold red]{safe_msg}[/bold red]", title="Configuration / Provider Error")
        )
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@generate_app.command("dockerfile")
def generate_dockerfile_cmd(
    language: str = typer.Argument(
        ..., help="Programming language or stack (e.g. python, nodejs, golang)."
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Target container application port."),
    base_image: Optional[str] = typer.Option(
        None, "--base-image", "-b", help="Base container image."
    ),
    instructions: Optional[str] = typer.Option(
        None, "--instructions", help="Custom generation instructions."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="File path to save the generated Dockerfile."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite of output file if it already exists."
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider (ollama or gemini)."
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override."),
    host: Optional[str] = typer.Option(None, "--host", help="Ollama host URL override."),
    timeout: Optional[float] = typer.Option(None, "--timeout", "-t", help="Timeout in seconds."),
    fallback: Optional[bool] = typer.Option(
        None, "--fallback/--no-fallback", help="Override automatic fallback."
    ),
):
    """Generate a validated, secure Dockerfile."""
    try:
        if provider and provider.lower() not in ("ollama", "gemini"):
            console.print(
                Panel(
                    f"[bold red]Unsupported provider: '{escape(provider)}'. Must be 'ollama' or 'gemini'.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        if port < 1 or port > 65535:
            console.print(
                Panel(
                    f"[bold red]Port must be between 1 and 65535, got {port}.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        generator = DockerfileGenerator()
        result = generator.generate(
            language=language,
            port=port,
            base_image=base_image,
            custom_instructions=instructions,
            output_path=output,
            force=force,
            provider_name=provider,
            model=model,
            host=host,
            timeout=timeout,
            allow_fallback=fallback,
        )
        subtitle = f"Provider: {escape(result.provider)} | Model: {escape(result.model)}"
        if result.fallback_used:
            subtitle += " [yellow](fallback)[/yellow]"
        console.print(
            Panel(
                escape(result.content),
                title=f"Generated Dockerfile ({escape(language)})",
                subtitle=subtitle,
            )
        )
        if result.warnings:
            console.print("[bold yellow]Heuristic Warnings:[/bold yellow]")
            for w in result.warnings:
                console.print(f"  [yellow]- {escape(w)}[/yellow]")
    except DevOpsAIError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Generation Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@generate_app.command("k8s")
def generate_k8s_cmd(
    app_name: str = typer.Argument(..., help="Application name."),
    image: str = typer.Option(
        ..., "--image", "-i", help="Container image (e.g. nginx:1.25, myapp:1.0)."
    ),
    port: int = typer.Option(8080, "--port", "-p", help="Container application port."),
    replicas: int = typer.Option(1, "--replicas", "-r", help="Number of deployment replicas."),
    service_type: str = typer.Option(
        "ClusterIP", "--service-type", "-s", help="Service type: ClusterIP, NodePort, LoadBalancer."
    ),
    namespace: str = typer.Option(
        "default", "--namespace", "-n", help="Target Kubernetes namespace."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="File path to save the generated manifest."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite of output file if it already exists."
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider (ollama or gemini)."
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override."),
    host: Optional[str] = typer.Option(None, "--host", help="Ollama host URL override."),
    timeout: Optional[float] = typer.Option(None, "--timeout", "-t", help="Timeout in seconds."),
    fallback: Optional[bool] = typer.Option(
        None, "--fallback/--no-fallback", help="Override automatic fallback."
    ),
):
    """Generate Kubernetes Deployment and Service manifests."""
    try:
        if provider and provider.lower() not in ("ollama", "gemini"):
            console.print(
                Panel(
                    f"[bold red]Unsupported provider: '{escape(provider)}'. Must be 'ollama' or 'gemini'.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        if port < 1 or port > 65535:
            console.print(
                Panel(
                    f"[bold red]Port must be between 1 and 65535, got {port}.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        if replicas < 1:
            console.print(
                Panel(
                    f"[bold red]Replicas must be an integer >= 1, got {replicas}.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        if service_type not in ("ClusterIP", "NodePort", "LoadBalancer"):
            console.print(
                Panel(
                    f"[bold red]Unsupported service type: '{escape(service_type)}'. Must be ClusterIP, NodePort, or LoadBalancer.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        generator = K8sManifestGenerator()
        result = generator.generate(
            app_name=app_name,
            image=image,
            port=port,
            replicas=replicas,
            namespace=namespace,
            service_type=service_type,  # type: ignore[arg-type]
            output_path=output,
            force=force,
            provider_name=provider,
            model=model,
            host=host,
            timeout=timeout,
            allow_fallback=fallback,
        )
        subtitle = f"Provider: {escape(result.provider)} | Model: {escape(result.model)}"
        if result.fallback_used:
            subtitle += " [yellow](fallback)[/yellow]"
        console.print(
            Panel(
                escape(result.content),
                title=f"Generated Kubernetes Manifests ({escape(app_name)})",
                subtitle=subtitle,
            )
        )
        if result.warnings:
            console.print("[bold yellow]Heuristic Warnings:[/bold yellow]")
            for w in result.warnings:
                console.print(f"  [yellow]- {escape(w)}[/yellow]")
    except DevOpsAIError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Generation Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@health_app.command("check")
def check_health(
    json_output: bool = typer.Option(False, "--json", help="Output health evaluation as JSON."),
):
    """Evaluate process liveness and readiness probes."""
    try:
        liveness = evaluate_liveness()
        status_code, readiness = evaluate_readiness()
        is_fatal = status_code == 503

        if json_output:
            payload = {"liveness": liveness, "readiness": readiness}
            print(json.dumps(payload, indent=2))
            if is_fatal:
                raise typer.Exit(code=1)
            return

        status_color = (
            "green"
            if readiness["status"] == "ready"
            else ("yellow" if readiness["status"] == "degraded" else "red")
        )
        storage_msg = readiness["checks"].get("filesystem", {}).get("message", "unknown")
        memory_msg = readiness["checks"].get("memory", {}).get("message", "unknown")
        console.print(
            Panel(
                f"Overall Status: [{status_color}]{escape(readiness['status'].upper())}[/{status_color}]\n"
                f"Uptime: {liveness['uptime_seconds']}s | Version: {escape(liveness['version'])}\n"
                f"Storage: {escape(storage_msg)}\n"
                f"Memory: {escape(memory_msg)}",
                title="[bold]DevOps AI Health Assessment[/bold]",
            )
        )
        if "provider" in readiness["checks"]:
            prov_info = readiness["checks"]["provider"]
            p_status = prov_info.get("status", "unknown")
            p_col = "green" if p_status == "ok" else "yellow"
            p_msg = prov_info.get("message", "")
            console.print(
                Panel(
                    f"Status: [{p_col}]{escape(p_status.upper())}[/{p_col}]\n"
                    f"Details: {escape(p_msg)}",
                    title="Default LLM Provider",
                )
            )

        if is_fatal:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        safe_msg = escape(_sanitize_cli_text(str(exc)))
        console.print(Panel(f"[bold red]Health check failed: {safe_msg}[/bold red]", title="Error"))
        raise typer.Exit(code=1)


@metrics_app.command("serve")
def serve_metrics(
    host: Optional[str] = typer.Option(
        None, "--host", "-h", help="Bind host for metrics HTTP server."
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Bind port for metrics HTTP server."
    ),
):
    """Start lightweight HTTP server exposing /metrics, /healthz, and /readyz."""
    try:
        settings = get_settings()
        bind_host = host or settings.metrics_host
        bind_port = port or settings.metrics_port

        if bind_port <= 0 or bind_port > 65535:
            console.print(
                Panel(
                    f"[bold red]Invalid port: {bind_port}. Must be between 1 and 65535.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        server = MetricsServer(host=bind_host, port=bind_port)
        server.start()
        console.print(
            Panel(
                f"[bold green]Serving telemetry endpoints:[/bold green]\n"
                f"  - Metrics:   http://{escape(bind_host)}:{bind_port}/metrics\n"
                f"  - Liveness:  http://{escape(bind_host)}:{bind_port}/healthz\n"
                f"  - Readiness: http://{escape(bind_host)}:{bind_port}/readyz\n\n"
                f"Press [bold cyan]Ctrl+C[/bold cyan] to stop.",
                title="Metrics Server",
            )
        )
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping metrics server...[/yellow]")
        finally:
            server.stop()
            console.print("[green]Metrics server stopped cleanly.[/green]")
    except typer.Exit:
        raise
    except Exception as exc:
        safe_msg = escape(_sanitize_cli_text(str(exc)))
        console.print(
            Panel(f"[bold red]Server failure: {safe_msg}[/bold red]", title="Server Error")
        )
        raise typer.Exit(code=1)


@metrics_app.command("dump")
def dump_metrics():
    """Dump current Prometheus metrics snapshot in exposition text format."""
    try:
        exposition = get_prometheus_exposition(force_refresh=True)
        print(exposition.decode("utf-8", errors="replace"), end="")
    except Exception as exc:
        safe_msg = escape(_sanitize_cli_text(str(exc)))
        console.print(
            Panel(f"[bold red]Failed to dump metrics: {safe_msg}[/bold red]", title="Metrics Error")
        )
        raise typer.Exit(code=1)


@monitor_app.command("docker")
def monitor_docker(
    all_containers: bool = typer.Option(
        True, "--all/--running", help="Show all containers or running only."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output status as JSON."),
    timeout: float = typer.Option(5.0, "--timeout", "-t", help="Timeout in seconds."),
):
    """Safely inspect Docker containers in a read-only manner."""
    try:
        monitor = DockerMonitor()
        status = monitor.get_status(all_containers=all_containers, timeout=timeout)

        if json_output:
            print(json.dumps(asdict(status), indent=2))
            if not status.is_available:
                raise typer.Exit(code=1)
            return

        if not status.is_available:
            err = escape(_sanitize_cli_text(status.error or "Docker daemon unreachable"))
            console.print(
                Panel(
                    f"[bold yellow]{err}[/bold yellow]",
                    title="Docker Status: [red]UNAVAILABLE[/red]",
                )
            )
            raise typer.Exit(code=1)

        console.print(
            Panel(
                f"Docker Daemon Version: [cyan]{escape(status.daemon_version or 'unknown')}[/cyan]\n"
                f"Total Containers: [bold]{len(status.containers)}[/bold]",
                title="Docker Status: [green]ONLINE[/green]",
            )
        )

        if status.containers:
            table = Table(title="Containers")
            table.add_column("Container ID", style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Image")
            table.add_column("State")
            table.add_column("Status")
            table.add_column("Ports")
            for c in status.containers:
                st_color = "green" if c.state.lower() == "running" else "yellow"
                table.add_row(
                    escape(c.container_id),
                    escape(c.name),
                    escape(c.image),
                    f"[{st_color}]{escape(c.state)}[/{st_color}]",
                    escape(c.status),
                    escape(c.ports),
                )
            console.print(table)
        else:
            console.print("[dim]No containers found.[/dim]")
    except typer.Exit:
        raise
    except Exception as exc:
        safe_msg = escape(_sanitize_cli_text(str(exc)))
        console.print(
            Panel(f"[bold red]Docker monitor error: {safe_msg}[/bold red]", title="Error")
        )
        raise typer.Exit(code=1)


@monitor_app.command("k8s")
def monitor_k8s(
    namespace: Optional[str] = typer.Option(
        None, "--namespace", "-n", help="Namespace filter (defaults to all namespaces)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output status as JSON."),
    timeout: float = typer.Option(5.0, "--timeout", "-t", help="Timeout in seconds."),
):
    """Safely inspect Kubernetes workloads in a read-only manner."""
    try:
        monitor = K8sMonitor()
        status = monitor.get_status(namespace=namespace, timeout=timeout)

        if json_output:
            print(json.dumps(asdict(status), indent=2))
            if not status.is_available:
                raise typer.Exit(code=1)
            return

        if not status.is_available:
            err = escape(_sanitize_cli_text(status.error or "Kubernetes cluster unreachable"))
            console.print(
                Panel(
                    f"[bold yellow]{err}[/bold yellow]",
                    title="Kubernetes Status: [red]UNAVAILABLE[/red]",
                )
            )
            raise typer.Exit(code=1)

        ns_text = f"Namespace: {escape(namespace)}" if namespace else "All Namespaces"
        console.print(
            Panel(
                f"Cluster Host: [cyan]{escape(status.cluster_host or 'configured')}[/cyan]\n"
                f"Scope: {ns_text}\n"
                f"Total Workloads: [bold]{len(status.workloads)}[/bold]",
                title="Kubernetes Status: [green]CONNECTED[/green]",
            )
        )

        if status.workloads:
            table = Table(title="Workloads")
            table.add_column("Kind", style="bold")
            table.add_column("Namespace", style="cyan")
            table.add_column("Name")
            table.add_column("Status")
            table.add_column("Created / Age")
            for w in status.workloads:
                table.add_row(
                    escape(w.kind),
                    escape(w.namespace),
                    escape(w.name),
                    escape(w.status),
                    escape(w.age),
                )
            console.print(table)
        else:
            console.print("[dim]No workloads found in scope.[/dim]")
    except typer.Exit:
        raise
    except Exception as exc:
        safe_msg = escape(_sanitize_cli_text(str(exc)))
        console.print(
            Panel(f"[bold red]Kubernetes monitor error: {safe_msg}[/bold red]", title="Error")
        )
        raise typer.Exit(code=1)


@app.command("log-analyze")
def log_analyze(
    file_path: Path = typer.Argument(
        ...,
        help="Path to the log file to parse and analyze.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    window: int = typer.Option(
        30, "--window", "-w", help="Burst detection sliding window in seconds."
    ),
    threshold: int = typer.Option(
        3, "--threshold", "-t", help="Minimum errors to trigger a burst alert."
    ),
    ml: bool = typer.Option(
        False, "--ml", help="Use Scikit-Learn IsolationForest for ML-based anomaly detection."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output analysis results as JSON."),
):
    """Analyze application or system logs for error bursts and volume anomalies."""
    try:
        if window <= 0:
            console.print(
                Panel(
                    f"[bold red]Window must be positive, got {window}.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)
        if threshold <= 0:
            console.print(
                Panel(
                    f"[bold red]Threshold must be positive, got {threshold}.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        parser = LogParser()
        entries = parser.parse_file(file_path)
        analyzer = LogAnalyzer()

        bursts = analyzer.detect_error_bursts(
            logs=entries, window_seconds=window, threshold=threshold
        )
        anomalies_result = analyzer.detect_anomalies(logs=entries)

        for _ in bursts:
            record_log_anomaly("burst")
        if anomalies_result.anomaly_count > 0:
            rec_type = (
                "ml_isolation_forest"
                if anomalies_result.method == "isolation_forest"
                else "statistical"
            )
            for _ in range(anomalies_result.anomaly_count):
                record_log_anomaly(rec_type)

        if json_output:
            bursts_data = [
                {
                    "has_burst": b.has_burst,
                    "error_count": b.error_count,
                    "window_start": b.window_start.isoformat() if b.window_start else None,
                    "window_end": b.window_end.isoformat() if b.window_end else None,
                    "message": b.message,
                }
                for b in bursts
            ]
            payload = {
                "file": str(file_path),
                "total_entries": len(entries),
                "bursts": bursts_data,
                "anomalies": anomalies_result.anomalies,
                "anomaly_count": anomalies_result.anomaly_count,
                "detection_method": anomalies_result.method,
            }
            print(json.dumps(payload, indent=2))
            return

        console.print(
            Panel(
                f"File: [cyan]{escape(str(file_path))}[/cyan]\n"
                f"Total Parsed Entries: [bold]{len(entries)}[/bold]\n"
                f"Error Bursts Detected: [bold]{len(bursts)}[/bold]\n"
                f"Volume Anomalies Detected: [bold]{anomalies_result.anomaly_count}[/bold] (Method: {escape(anomalies_result.method)})",
                title="Log Analysis Results",
            )
        )

        if bursts:
            burst_table = Table(title="Error Bursts")
            burst_table.add_column("Window Start", style="cyan")
            burst_table.add_column("Window End", style="cyan")
            burst_table.add_column("Error Count", style="bold red")
            burst_table.add_column("Message")
            for b in bursts:
                w_start = b.window_start.isoformat() if b.window_start else "-"
                w_end = b.window_end.isoformat() if b.window_end else "-"
                burst_table.add_row(
                    escape(w_start),
                    escape(w_end),
                    str(b.error_count),
                    escape(b.message),
                )
            console.print(burst_table)

        if anomalies_result.anomalies:
            anom_table = Table(title="Operational Anomalies")
            anom_table.add_column("Timestamp", style="cyan")
            anom_table.add_column("Level", style="bold")
            anom_table.add_column("Message")
            for a in anomalies_result.anomalies:
                lvl = str(a.get("level", "UNKNOWN"))
                lvl_col = "red" if lvl in ("ERROR", "CRITICAL") else "yellow"
                anom_table.add_row(
                    escape(str(a.get("timestamp", "-"))),
                    f"[{lvl_col}]{escape(lvl)}[/{lvl_col}]",
                    escape(str(a.get("message", ""))[:100]),
                )
            console.print(anom_table)

    except DevOpsAIError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Log Analysis Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


# =====================================================================
# Incident Investigation & Lifecycle Management Subcommands
# =====================================================================


@incident_app.command("analyze")
def incident_analyze(
    incident_id: str = typer.Argument(..., help="Unique incident identifier to analyze."),
    force_tier2: bool = typer.Option(
        False,
        "--force-tier2",
        help="Force Tier 2 AI investigation regardless of Tier 1 confidence.",
    ),
    skip_tier2: bool = typer.Option(
        False,
        "--skip-tier2",
        help="Skip Tier 2 AI investigation and retain Tier 1 deterministic results only.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit raw JSON representation of the analyzed incident."
    ),
):
    """Perform two-tier root cause analysis on an incident."""
    try:
        if json_output:
            logging.getLogger("devops_ai").setLevel(logging.WARNING)

        repo = IncidentRepository()
        incident = repo.get_or_raise(incident_id)
        analyzer = IncidentAnalyzer()
        analyzed = analyzer.analyze(incident, force_tier2=force_tier2, skip_tier2=skip_tier2)
        saved = repo.save(analyzed)

        if json_output:
            print(IncidentReporter.render_json(saved))
        else:
            IncidentReporter.render_terminal(saved, console=console)
    except IncidentNotFoundError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Incident Not Found"))
        raise typer.Exit(code=1)
    except (RepositoryLockError, ConcurrencyConflictError, DevOpsAIError) as e:
        safe_msg = escape(_sanitize_cli_text(e.message if hasattr(e, "message") else str(e)))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Analysis Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@incident_app.command("list")
def incident_list(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (DETECTED, CORRELATED, INVESTIGATING, ANALYZED, RESOLVED, CLOSED).",
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", help="Filter by severity (INFO, LOW, MEDIUM, HIGH, CRITICAL)."
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="Filter by category (INFRASTRUCTURE, CONTAINER, APPLICATION, NETWORK, PROVIDER, UNKNOWN).",
    ),
    active_only: bool = typer.Option(
        False, "--active-only", help="Filter for active incidents (excluding RESOLVED and CLOSED)."
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of incidents to return."),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON list of incidents."),
):
    """List operational incidents with optional filtering."""
    try:
        status_filter: Optional[IncidentStatus] = None
        if status:
            try:
                status_filter = IncidentStatus(status.strip().upper())
            except ValueError:
                console.print(
                    Panel(
                        f"[bold red]Invalid status: '{escape(status)}'. Must be one of: "
                        f"{', '.join(s.value for s in IncidentStatus)}[/bold red]",
                        title="Input Error",
                    )
                )
                raise typer.Exit(code=2)

        severity_filter: Optional[Severity] = None
        if severity:
            try:
                severity_filter = Severity(severity.strip().upper())
            except ValueError:
                console.print(
                    Panel(
                        f"[bold red]Invalid severity: '{escape(severity)}'. Must be one of: "
                        f"{', '.join(s.value for s in Severity)}[/bold red]",
                        title="Input Error",
                    )
                )
                raise typer.Exit(code=2)

        category_filter: Optional[Category] = None
        if category:
            try:
                category_filter = Category(category.strip().upper())
            except ValueError:
                console.print(
                    Panel(
                        f"[bold red]Invalid category: '{escape(category)}'. Must be one of: "
                        f"{', '.join(c.value for c in Category)}[/bold red]",
                        title="Input Error",
                    )
                )
                raise typer.Exit(code=2)

        repo = IncidentRepository()
        incidents = repo.list_all(
            status=status_filter,
            severity=severity_filter,
            active_only=active_only,
            limit=limit,
        )

        if category_filter:
            incidents = [inc for inc in incidents if inc.category == category_filter]

        if json_output:
            raw_list = [json.loads(IncidentReporter.render_json(inc)) for inc in incidents]
            print(json.dumps(raw_list, indent=2))
            return

        if not incidents:
            console.print(
                Panel(
                    "[dim]No incidents matching the specified criteria were found.[/dim]",
                    title="DevOps AI Incidents",
                    border_style="cyan",
                )
            )
            return

        table = Table(
            title="DevOps AI Operational Incidents", show_header=True, header_style="bold"
        )
        table.add_column("Incident ID", style="bold cyan", no_wrap=True)
        table.add_column("Severity")
        table.add_column("Status")
        table.add_column("Category", style="magenta")
        table.add_column("Title", style="white")
        table.add_column("Components", style="dim")
        table.add_column("Updated (UTC)", style="dim")

        for inc in incidents:
            sev_col = (
                "bold red"
                if inc.severity == Severity.CRITICAL
                else ("red" if inc.severity == Severity.HIGH else "yellow")
            )
            stat_col = (
                "green"
                if inc.status == IncidentStatus.RESOLVED
                else ("dim green" if inc.status == IncidentStatus.CLOSED else "yellow")
            )
            comps = ", ".join(inc.components_affected) or "None"
            table.add_row(
                escape(inc.incident_id),
                f"[{sev_col}]{escape(inc.severity.value)}[/{sev_col}]",
                f"[{stat_col}]{escape(inc.status.value)}[/{stat_col}]",
                escape(inc.category.value),
                escape(_sanitize_cli_text(inc.title)),
                escape(comps),
                escape(inc.updated_at.strftime("%Y-%m-%d %H:%M:%S")),
            )

        console.print(table)
    except (RepositoryLockError, DevOpsAIError) as e:
        safe_msg = escape(_sanitize_cli_text(e.message if hasattr(e, "message") else str(e)))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="List Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@incident_app.command("show")
def incident_show(
    incident_id: str = typer.Argument(..., help="Unique incident identifier to view."),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON representation."),
):
    """Display comprehensive details, root cause, and recommendations for an incident."""
    try:
        repo = IncidentRepository()
        incident = repo.get_or_raise(incident_id)

        if json_output:
            print(IncidentReporter.render_json(incident))
        else:
            IncidentReporter.render_terminal(incident, console=console)
    except IncidentNotFoundError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Incident Not Found"))
        raise typer.Exit(code=1)
    except (RepositoryLockError, DevOpsAIError) as e:
        safe_msg = escape(_sanitize_cli_text(e.message if hasattr(e, "message") else str(e)))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Operation Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@incident_app.command("resolve")
def incident_resolve(
    incident_id: str = typer.Argument(..., help="Unique incident identifier to resolve."),
    notes: Optional[str] = typer.Option(
        None, "--notes", "-n", help="Resolution notes or mitigation summary."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit raw JSON representation of the resolved incident."
    ),
):
    """Mark an incident as RESOLVED with resolution notes and timeline entry."""
    try:
        repo = IncidentRepository()
        incident = repo.get_or_raise(incident_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = now
        if notes:
            incident.resolution_notes = notes

        incident.timeline.append(
            TimelineEntry(
                timestamp=now,
                source=AlertSource.MANUAL,
                component=incident.components_affected[0]
                if incident.components_affected
                else "system",
                event=f"Incident resolved by operator: {notes or 'No resolution notes provided'}",
                severity=Severity.INFO,
            )
        )
        saved = repo.save(incident)

        if json_output:
            print(IncidentReporter.render_json(saved))
        else:
            console.print(
                Panel(
                    f"[bold green]Incident '{escape(saved.incident_id)}' marked as RESOLVED.[/bold green]\n"
                    f"Notes: {escape(_sanitize_cli_text(notes or 'None'))}",
                    title="Incident Resolved",
                    border_style="green",
                )
            )
    except IncidentNotFoundError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Incident Not Found"))
        raise typer.Exit(code=1)
    except (RepositoryLockError, ConcurrencyConflictError, DevOpsAIError) as e:
        safe_msg = escape(_sanitize_cli_text(e.message if hasattr(e, "message") else str(e)))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Resolve Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@incident_app.command("close")
def incident_close(
    incident_id: str = typer.Argument(..., help="Unique incident identifier to close."),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Post-incident closing notes."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit raw JSON representation of the closed incident."
    ),
):
    """Mark an incident as CLOSED with closing notes and timeline entry."""
    try:
        repo = IncidentRepository()
        incident = repo.get_or_raise(incident_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        incident.status = IncidentStatus.CLOSED
        incident.closed_at = now
        if not incident.resolved_at:
            incident.resolved_at = now

        if notes:
            if incident.resolution_notes:
                incident.resolution_notes = f"{incident.resolution_notes}\n[Closed]: {notes}"
            else:
                incident.resolution_notes = f"[Closed]: {notes}"

        incident.timeline.append(
            TimelineEntry(
                timestamp=now,
                source=AlertSource.MANUAL,
                component=incident.components_affected[0]
                if incident.components_affected
                else "system",
                event=f"Incident closed by operator: {notes or 'No closing notes provided'}",
                severity=Severity.INFO,
            )
        )
        saved = repo.save(incident)

        if json_output:
            print(IncidentReporter.render_json(saved))
        else:
            console.print(
                Panel(
                    f"[bold green]Incident '{escape(saved.incident_id)}' marked as CLOSED.[/bold green]\n"
                    f"Notes: {escape(_sanitize_cli_text(notes or 'None'))}",
                    title="Incident Closed",
                    border_style="green",
                )
            )
    except IncidentNotFoundError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Incident Not Found"))
        raise typer.Exit(code=1)
    except (RepositoryLockError, ConcurrencyConflictError, DevOpsAIError) as e:
        safe_msg = escape(_sanitize_cli_text(e.message if hasattr(e, "message") else str(e)))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Close Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@incident_app.command("report")
def incident_report(
    incident_id: str = typer.Argument(..., help="Unique incident identifier to export report for."),
    format_type: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown, json, or terminal."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="File path to write report to (defaults to stdout/console)."
    ),
):
    """Generate and export a structured post-mortem report for an incident."""
    try:
        fmt = format_type.strip().lower()
        if fmt not in ("markdown", "md", "json", "terminal"):
            console.print(
                Panel(
                    f"[bold red]Unsupported format: '{escape(format_type)}'. Must be 'markdown', 'json', or 'terminal'.[/bold red]",
                    title="Input Error",
                )
            )
            raise typer.Exit(code=2)

        repo = IncidentRepository()
        incident = repo.get_or_raise(incident_id)

        if output:
            IncidentReporter.export_report(
                incident, format=fmt, output_path=output, console=console
            )
            console.print(
                Panel(
                    f"[bold green]Post-mortem report exported successfully.[/bold green]\n"
                    f"Format: [cyan]{escape(fmt)}[/cyan]\n"
                    f"Destination: [cyan]{escape(str(output))}[/cyan]",
                    title="Report Exported",
                    border_style="green",
                )
            )
        else:
            if fmt in ("markdown", "md"):
                print(IncidentReporter.render_markdown(incident))
            elif fmt == "json":
                print(IncidentReporter.render_json(incident))
            elif fmt == "terminal":
                IncidentReporter.render_terminal(incident, console=console)
    except IncidentNotFoundError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Incident Not Found"))
        raise typer.Exit(code=1)
    except (RepositoryLockError, DevOpsAIError) as e:
        safe_msg = escape(_sanitize_cli_text(e.message if hasattr(e, "message") else str(e)))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Report Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


@incident_app.command("rca")
def incident_rca(
    incident_id: str = typer.Argument(..., help="Unique incident identifier to diagnose."),
    force_tier2: bool = typer.Option(
        False,
        "--force-tier2",
        help="Force Tier 2 AI investigation regardless of Tier 1 confidence.",
    ),
    skip_tier2: bool = typer.Option(
        False,
        "--skip-tier2",
        help="Skip Tier 2 AI investigation and use Tier 1 deterministic engine only.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit root cause and remediation recommendations as JSON."
    ),
):
    """Run targeted root-cause analysis and view diagnosis with remediation advisory."""
    try:
        if json_output:
            logging.getLogger("devops_ai").setLevel(logging.WARNING)

        repo = IncidentRepository()
        incident = repo.get_or_raise(incident_id)
        analyzer = IncidentAnalyzer()
        analyzed = analyzer.analyze(incident, force_tier2=force_tier2, skip_tier2=skip_tier2)
        saved = repo.save(analyzed)

        if json_output:
            rca_payload = {
                "incident_id": saved.incident_id,
                "status": saved.status.value,
                "root_cause": saved.root_cause.model_dump(mode="json")
                if saved.root_cause
                else None,
                "alternative_hypotheses": [
                    h.model_dump(mode="json") for h in saved.alternative_hypotheses
                ],
                "recommendations": [r.model_dump(mode="json") for r in saved.recommendations],
            }
            print(json.dumps(rca_payload, indent=2))
        else:
            if saved.root_cause:
                rc = saved.root_cause
                tier_badge = (
                    "Tier 1 (Deterministic)"
                    if rc.tier == "tier_1_deterministic"
                    else "Tier 2 (AI-Assisted)"
                )
                stat_str = (
                    "[bold green]CONFIRMED[/bold green]"
                    if rc.is_confirmed
                    else (
                        "[bold red]UNSUPPORTED[/bold red]"
                        if rc.is_unsupported
                        else "[yellow]UNCONFIRMED[/yellow]"
                    )
                )
                console.print(
                    Panel(
                        f"Incident ID: [bold cyan]{escape(saved.incident_id)}[/bold cyan]\n"
                        f"Root Cause: [bold yellow]{escape(_sanitize_cli_text(rc.title))}[/bold yellow]\n"
                        f"Engine: [cyan]{tier_badge}[/cyan] | Status: {stat_str} | Confidence: [bold]{rc.confidence:.0%}[/bold]\n\n"
                        f"{escape(_sanitize_cli_text(rc.description))}",
                        title="[bold yellow]Root Cause Diagnosis[/bold yellow]",
                        border_style="yellow",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"No conclusive root cause identified for incident '{escape(saved.incident_id)}'.",
                        title="Root Cause Diagnosis",
                        border_style="dim",
                    )
                )

            if saved.recommendations:
                rec_table = Table(
                    title="Remediation Advisory (ADVISORY ONLY — Human Execution Required)",
                    show_header=True,
                    header_style="bold",
                )
                rec_table.add_column("Action ID", style="dim", width=12)
                rec_table.add_column("Risk", width=12)
                rec_table.add_column("Command / Action", style="white")
                rec_table.add_column("Verification Step", style="green")

                for rec in saved.recommendations:
                    risk_style = (
                        "bold green"
                        if rec.risk_level.value == "SAFE"
                        else ("bold yellow" if rec.risk_level.value == "MODERATE" else "bold red")
                    )
                    cmd_str = (
                        f"[bold]{escape(_sanitize_cli_text(rec.command))}[/bold]"
                        if rec.command
                        else f"[italic]{escape(_sanitize_cli_text(rec.title))}[/italic]"
                    )
                    rec_table.add_row(
                        escape(rec.action_id),
                        f"[{risk_style}]{escape(rec.risk_level.value)}[/{risk_style}]",
                        cmd_str,
                        escape(_sanitize_cli_text(rec.verification_step)),
                    )
                console.print(rec_table)
    except IncidentNotFoundError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Incident Not Found"))
        raise typer.Exit(code=1)
    except (RepositoryLockError, ConcurrencyConflictError, DevOpsAIError) as e:
        safe_msg = escape(_sanitize_cli_text(e.message if hasattr(e, "message") else str(e)))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="RCA Failed"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error")
        )
        raise typer.Exit(code=1)


# =====================================================================
# RAG Subsystem CLI Commands (Phase 5 Commit 4)
# =====================================================================


@rag_app.command("evaluate")
def evaluate_rag(
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Path to JSON or YAML evaluation dataset file (defaults to built-in synthetic benchmark).",
    ),
    k_cutoffs: str = typer.Option(
        "1,3,5",
        "--k",
        "-k",
        help="Comma-separated list of top-K rank cutoffs to evaluate (e.g. '1,3,5').",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write full JSON evaluation report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw evaluation report as JSON to stdout.",
    ),
    min_precision: Optional[float] = typer.Option(
        None,
        "--min-precision",
        help="Minimum acceptable mean Precision@K threshold (fails with exit code 1 if violated).",
    ),
    min_recall: Optional[float] = typer.Option(
        None,
        "--min-recall",
        help="Minimum acceptable mean Recall@K threshold (fails with exit code 1 if violated).",
    ),
    min_mrr: Optional[float] = typer.Option(
        None,
        "--min-mrr",
        help="Minimum acceptable Mean Reciprocal Rank threshold (fails with exit code 1 if violated).",
    ),
    min_ndcg: Optional[float] = typer.Option(
        None,
        "--min-ndcg",
        help="Minimum acceptable mean nDCG@K threshold (fails with exit code 1 if violated).",
    ),
    max_latency_ms: Optional[float] = typer.Option(
        None,
        "--max-latency-ms",
        help="Maximum allowable p95 total retrieval latency in milliseconds (fails with exit code 1 if exceeded).",
    ),
):
    """Evaluate RAG semantic retrieval quality, answer grounding, and query latency."""
    setup_logging(level="WARNING")
    settings = get_settings()

    # 1. Parse top-K cutoffs
    try:
        k_values = [int(x.strip()) for x in k_cutoffs.split(",") if x.strip()]
        k_values = [k for k in k_values if k > 0]
        if not k_values:
            k_values = [1, 3, 5]
    except Exception:
        console.print(
            "[bold red]Invalid --k format. Expected comma-separated integers, e.g. '1,3,5'[/bold red]"
        )
        raise typer.Exit(code=1)

    # 2. Build quality thresholds map
    thresholds: Dict[str, float] = {}
    if min_precision is not None:
        thresholds["min_precision"] = float(min_precision)
    if min_recall is not None:
        thresholds["min_recall"] = float(min_recall)
    if min_mrr is not None:
        thresholds["min_mrr"] = float(min_mrr)
    if min_ndcg is not None:
        thresholds["min_ndcg"] = float(min_ndcg)
    if max_latency_ms is not None:
        thresholds["max_latency_ms"] = float(max_latency_ms)

    # 3. Load dataset
    try:
        if dataset_path:
            dataset = load_evaluation_dataset(
                dataset_path, max_queries=settings.rag_eval_max_queries
            )
        else:
            dataset = get_builtin_evaluation_dataset()
    except DatasetValidationError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Dataset Validation Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Failed to load evaluation dataset: {safe_msg}[/bold red]",
                title="Dataset Error",
            )
        )
        raise typer.Exit(code=1)

    # 4. Execute evaluation
    try:
        evaluator = RAGEvaluator(settings=settings)
        report = evaluator.evaluate_dataset(
            dataset=dataset,
            k_values=k_values,
            quality_thresholds=thresholds,
            raise_on_threshold_failure=False,
        )
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Evaluation failed: {safe_msg}[/bold red]", title="Evaluation Error")
        )
        raise typer.Exit(code=1)

    # 5. Save output if requested
    if output_path:
        try:
            output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            safe_msg = escape(_sanitize_cli_text(str(e)))
            console.print(
                Panel(
                    f"[bold red]Failed to write output file: {safe_msg}[/bold red]",
                    title="IO Error",
                )
            )
            raise typer.Exit(code=1)

    # 6. Output format: JSON or Rich Console
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        if not report.passed_quality_gates:
            raise typer.Exit(code=1)
        return

    # Rich human-readable output
    console.print(
        Panel(
            f"[bold]Dataset ID:[/bold] {escape(report.dataset_id)}\n"
            f"[bold]Dataset Name:[/bold] {escape(report.dataset_name)}\n"
            f"[bold]Total Queries:[/bold] {report.total_queries}\n"
            f"[bold]Evaluated Cutoffs (K):[/bold] {report.k_values}",
            title="[bold cyan]RAG Retrieval Evaluation Benchmark[/bold cyan]",
        )
    )

    # Table 1: Retrieval Quality Metrics
    retrieval_table = Table(title="Retrieval Quality Metrics", show_header=True)
    retrieval_table.add_column("Metric", style="bold cyan")
    for k in report.k_values:
        retrieval_table.add_column(f"k={k}", justify="right")
    retrieval_table.add_column("Threshold", justify="right")
    retrieval_table.add_column("Status", justify="center")

    max_k = max(report.k_values)

    # Precision row
    p_cells = [
        f"{report.mean_retrieval_metrics.precision_at_k.get(k, 0.0):.4f}" for k in report.k_values
    ]
    p_thresh = f">= {thresholds['min_precision']:.4f}" if "min_precision" in thresholds else "-"
    p_status = (
        "[bold green]PASS[/bold green]"
        if (
            "min_precision" not in thresholds
            or report.mean_retrieval_metrics.precision_at_k.get(max_k, 0.0)
            >= thresholds["min_precision"]
        )
        else "[bold red]FAIL[/bold red]"
    )
    retrieval_table.add_row("Precision@K", *p_cells, p_thresh, p_status)

    # Recall row
    r_cells = [
        f"{report.mean_retrieval_metrics.recall_at_k.get(k, 0.0):.4f}" for k in report.k_values
    ]
    r_thresh = f">= {thresholds['min_recall']:.4f}" if "min_recall" in thresholds else "-"
    r_status = (
        "[bold green]PASS[/bold green]"
        if (
            "min_recall" not in thresholds
            or report.mean_retrieval_metrics.recall_at_k.get(max_k, 0.0) >= thresholds["min_recall"]
        )
        else "[bold red]FAIL[/bold red]"
    )
    retrieval_table.add_row("Recall@K", *r_cells, r_thresh, r_status)

    # MRR row
    mrr_cells = [f"{report.mean_retrieval_metrics.mrr:.4f}"] + ["-" for _ in report.k_values[1:]]
    mrr_thresh = f">= {thresholds['min_mrr']:.4f}" if "min_mrr" in thresholds else "-"
    mrr_status = (
        "[bold green]PASS[/bold green]"
        if (
            "min_mrr" not in thresholds
            or report.mean_retrieval_metrics.mrr >= thresholds["min_mrr"]
        )
        else "[bold red]FAIL[/bold red]"
    )
    retrieval_table.add_row("MRR", *mrr_cells, mrr_thresh, mrr_status)

    # nDCG row
    ndcg_cells = [
        f"{report.mean_retrieval_metrics.ndcg_at_k.get(k, 0.0):.4f}" for k in report.k_values
    ]
    ndcg_thresh = f">= {thresholds['min_ndcg']:.4f}" if "min_ndcg" in thresholds else "-"
    ndcg_status = (
        "[bold green]PASS[/bold green]"
        if (
            "min_ndcg" not in thresholds
            or report.mean_retrieval_metrics.ndcg_at_k.get(max_k, 0.0) >= thresholds["min_ndcg"]
        )
        else "[bold red]FAIL[/bold red]"
    )
    retrieval_table.add_row("nDCG@K", *ndcg_cells, ndcg_thresh, ndcg_status)

    console.print(retrieval_table)

    # Table 2: Latency & Observability
    lat_table = Table(title="Retrieval Latency Observability (ms)", show_header=True)
    lat_table.add_column("Pipeline Stage", style="bold cyan")
    lat_table.add_column("Mean", justify="right")
    lat_table.add_column("p50", justify="right")
    lat_table.add_column("p90", justify="right")
    lat_table.add_column("p95", justify="right")
    lat_table.add_column("p99", justify="right")
    lat_table.add_column("Max", justify="right")

    for name, dist in [
        ("Query Embedding", report.latency_summary.embedding_latency),
        ("Vector Search", report.latency_summary.search_latency),
        ("Total Retrieval", report.latency_summary.total_latency),
    ]:
        lat_table.add_row(
            name,
            f"{dist.mean:.2f} ms",
            f"{dist.p50:.2f} ms",
            f"{dist.p90:.2f} ms",
            f"{dist.p95:.2f} ms",
            f"{dist.p99:.2f} ms",
            f"{dist.max:.2f} ms",
        )
    console.print(lat_table)

    # Table 3: Groundedness & Citations (if present)
    if report.mean_groundedness_metrics:
        gm = report.mean_groundedness_metrics
        ground_table = Table(
            title="Answer Grounding & Citation Verification (Proxy Heuristics)", show_header=True
        )
        ground_table.add_column("Metric / Property", style="bold cyan")
        ground_table.add_column("Value", justify="right")

        ground_table.add_row("Citation Coverage", f"{gm.citation_coverage * 100:.1f}%")
        ground_table.add_row(
            "Evidence Overlap Score (proxy)", f"{gm.evidence_overlap_score * 100:.1f}%"
        )
        ground_table.add_row("Total Claims Evaluated", str(gm.total_claims))
        ground_table.add_row("Supported Claims", f"[green]{gm.supported_claims}[/green]")
        ground_table.add_row(
            "Unsupported Claims",
            f"[yellow]{gm.unsupported_claims}[/yellow]" if gm.unsupported_claims > 0 else "0",
        )
        if gm.hallucinated_citations:
            ground_table.add_row(
                "Hallucinated Citations",
                f"[red]{escape(', '.join(gm.hallucinated_citations))}[/red]",
            )
        else:
            ground_table.add_row("Hallucinated Citations", "[green]None[/green]")
        console.print(ground_table)

    # Final Quality Gate Summary
    if report.passed_quality_gates:
        console.print(
            Panel(
                "[bold green]✓ All quality gate thresholds satisfied (Exit Code: 0)[/bold green]",
                title="Quality Gates",
            )
        )
    else:
        failed_checks = [k for k, v in report.threshold_checks.items() if not v]
        console.print(
            Panel(
                "[bold red]✗ Quality gate threshold violation(s):[/bold red]\n"
                + "\n".join(f"  - {check}" for check in failed_checks)
                + "\n\n[yellow]Failing with exit code 1[/yellow]",
                title="[bold red]Quality Gates Failed[/bold red]",
            )
        )
        raise typer.Exit(code=1)


# =====================================================================
# FinOps Subsystem CLI Commands (Phase 6 Commit 1)
# =====================================================================


@finops_app.command("analyze")
def analyze_cloud_costs(
    billing_file: Optional[Path] = typer.Option(
        None,
        "--billing-file",
        "--file",
        "-b",
        help="Path to CSV or JSON billing export file (defaults to built-in synthetic benchmark).",
    ),
    service: Optional[str] = typer.Option(
        None,
        "--service",
        "-s",
        help="Filter expenditure by cloud service name (e.g. AmazonEC2, BigQuery).",
    ),
    account: Optional[str] = typer.Option(
        None,
        "--account",
        "-a",
        help="Filter expenditure by cloud account/subscription/project ID.",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="Filter expenditure by cloud region (e.g. us-east-1, eastus).",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Filter expenditure by start date (YYYY-MM-DD).",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="Filter expenditure by end date (YYYY-MM-DD).",
    ),
    group_by: str = typer.Option(
        "service",
        "--group-by",
        "-g",
        help="Dimension to group costs by in table display: 'service', 'account', 'region', or 'date'.",
    ),
    allowed_root: Optional[Path] = typer.Option(
        None,
        "--allowed-root",
        help="Permitted filesystem root directory to enforce path traversal safety.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save full JSON cost summary report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw cost summary report as JSON to stdout.",
    ),
):
    """Analyze cloud expenditure with multi-dimensional breakdowns and summaries."""
    setup_logging(level="WARNING")
    settings = get_settings()
    op_start = time.monotonic()

    # 1. Validate date filters if provided
    parsed_start_date: Optional[datetime.date] = None
    if start_date:
        try:
            parsed_start_date = datetime.date.fromisoformat(start_date.strip())
        except ValueError:
            record_finops_operation("analyze", "failure")
            console.print(
                Panel(
                    f"[bold red]Invalid --start-date format '{start_date}'. Expected YYYY-MM-DD.[/bold red]",
                    title="Invalid Date Format",
                )
            )
            raise typer.Exit(code=1)

    parsed_end_date: Optional[datetime.date] = None
    if end_date:
        try:
            parsed_end_date = datetime.date.fromisoformat(end_date.strip())
        except ValueError:
            record_finops_operation("analyze", "failure")
            console.print(
                Panel(
                    f"[bold red]Invalid --end-date format '{end_date}'. Expected YYYY-MM-DD.[/bold red]",
                    title="Invalid Date Format",
                )
            )
            raise typer.Exit(code=1)

    if parsed_start_date and parsed_end_date and parsed_start_date > parsed_end_date:
        record_finops_operation("analyze", "failure")
        console.print(
            Panel(
                f"[bold red]--start-date ({parsed_start_date}) cannot be later than --end-date ({parsed_end_date}).[/bold red]",
                title="Invalid Date Range",
            )
        )
        raise typer.Exit(code=1)

    # 2. Load billing records
    try:
        if billing_file:
            records, detected_schema = load_billing_records(
                path=billing_file,
                allowed_root=allowed_root,
                max_records=settings.finops_max_records,
                max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
            )
        else:
            records = get_synthetic_billing_dataset()
            detected_schema = "SYNTHETIC_BENCHMARK"
    except BillingDataError as e:
        record_finops_operation("analyze", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Billing Data Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("analyze", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Failed to load billing records: {safe_msg}[/bold red]",
                title="Import Error",
            )
        )
        raise typer.Exit(code=1)

    # 3. Filter records
    analyzer = CostAnalyzer()
    filtered_records = analyzer.filter_records(
        records=records,
        service=service,
        account=account,
        region=region,
        start_date=parsed_start_date,
        end_date=parsed_end_date,
    )

    # 4. Compute cost summary
    try:
        summary = analyzer.summarize(
            records=filtered_records,
            base_currency=settings.finops_default_currency,
        )
    except CurrencyMismatchError as e:
        record_finops_operation("analyze", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Currency Mismatch Error"))
        raise typer.Exit(code=1)
    except FinOpsError as e:
        record_finops_operation("analyze", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="FinOps Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("analyze", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Analysis failed: {safe_msg}[/bold red]", title="Analysis Error")
        )
        raise typer.Exit(code=1)

    # 5. Save output file if requested
    if output_path:
        try:
            output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            record_finops_operation("analyze", "failure")
            safe_msg = escape(_sanitize_cli_text(str(e)))
            console.print(
                Panel(
                    f"[bold red]Failed to write output file: {safe_msg}[/bold red]",
                    title="IO Error",
                )
            )
            raise typer.Exit(code=1)

    # 6. Output format: JSON or Rich Console
    if json_output:
        op_duration = max(0.0, time.monotonic() - op_start)
        record_finops_operation("analyze", "success", duration_seconds=op_duration)
        typer.echo(summary.model_dump_json(indent=2))
        return

    # Rich human-readable output
    def _fmt_money(val: float) -> str:
        if val < 0:
            return f"-${abs(val):,.2f}"
        return f"${val:,.2f}"

    start_str = (
        summary.period_start.strftime("%Y-%m-%d %H:%M UTC") if summary.period_start else "N/A"
    )
    end_str = summary.period_end.strftime("%Y-%m-%d %H:%M UTC") if summary.period_end else "N/A"

    console.print(
        Panel(
            f"[bold]Detected Schema:[/bold] {escape(detected_schema)}\n"
            f"[bold]Net Spend:[/bold] [bold green]{_fmt_money(summary.total_cost)} {summary.currency}[/bold green]\n"
            f"[bold]Gross Charges:[/bold] {_fmt_money(summary.gross_cost)} {summary.currency}\n"
            f"[bold]Credits & Refunds:[/bold] [bold cyan]{_fmt_money(summary.credit_cost)} {summary.currency}[/bold cyan]\n"
            f"[bold]Processed Records:[/bold] {summary.total_records}\n"
            f"[bold]Billing Window:[/bold] {start_str} -> {end_str}",
            title="[bold cyan]Cloud FinOps Cost Analysis[/bold cyan]",
        )
    )

    # Dimensional Breakdown Table based on --group-by
    norm_group = group_by.strip().lower()
    if norm_group == "account":
        table_title = "Cost by Cloud Account / Subscription"
        breakdown_data = summary.account_costs
        dim_col = "Account ID"
    elif norm_group == "region":
        table_title = "Cost by Cloud Region"
        breakdown_data = summary.region_costs
        dim_col = "Region"
    elif norm_group == "date":
        table_title = "Daily Spend Breakdown"
        breakdown_data = summary.daily_costs
        dim_col = "Date"
    else:
        table_title = "Cost by Cloud Service"
        breakdown_data = summary.service_costs
        dim_col = "Service Name"

    cost_table = Table(title=table_title, show_header=True)
    cost_table.add_column(dim_col, style="bold cyan")
    cost_table.add_column("Spend Amount", justify="right")
    cost_table.add_column("Share (%)", justify="right")

    for k, v in breakdown_data.items():
        pct = (v / summary.total_cost * 100.0) if summary.total_cost > 0 else 0.0
        cost_table.add_row(escape(k), _fmt_money(v), f"{pct:.1f}%")

    console.print(cost_table)

    # Top Spending Resources Table
    if summary.top_resources:
        res_table = Table(title="Top Spending Resources", show_header=True)
        res_table.add_column("Resource Identifier", style="bold cyan")
        res_table.add_column("Service", style="green")
        res_table.add_column("Spend Amount", justify="right")
        res_table.add_column("Share (%)", justify="right")

        for r_item in summary.top_resources:
            r_name = (
                f"{r_item['resource_id']} ({r_item['resource_name']})"
                if r_item.get("resource_name")
                else r_item["resource_id"]
            )
            res_table.add_row(
                escape(r_name),
                escape(r_item["service_name"]),
                _fmt_money(r_item["cost"]),
                f"{r_item['percentage']:.1f}%",
            )
        console.print(res_table)

    op_duration = max(0.0, time.monotonic() - op_start)
    record_finops_operation("analyze", "success", duration_seconds=op_duration)


@finops_app.command("optimize")
def optimize_cloud_resources(
    inventory_file: Optional[Path] = typer.Option(
        None,
        "--inventory-file",
        "--file",
        "-f",
        help="Path to JSON or CSV cloud resource inventory file (defaults to built-in synthetic benchmark).",
    ),
    utilization_file: Optional[Path] = typer.Option(
        None,
        "--utilization-file",
        "-u",
        help="Path to JSON or CSV resource utilization metrics file.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Filter by cloud provider: 'aws', 'azure', 'gcp', 'kubernetes'.",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by resource category: 'compute', 'storage', 'database', 'network'.",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="Filter by cloud region (e.g. us-east-1, eastus).",
    ),
    risk: Optional[str] = typer.Option(
        None,
        "--risk",
        help="Maximum risk ceiling to include: 'LOW', 'MEDIUM', 'HIGH'.",
    ),
    min_savings: float = typer.Option(
        0.0,
        "--min-savings",
        help="Minimum estimated monthly savings threshold to include a recommendation.",
    ),
    allowed_root: Optional[Path] = typer.Option(
        None,
        "--allowed-root",
        help="Permitted filesystem root directory to enforce path traversal safety.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save full JSON optimization summary report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw optimization summary report as JSON to stdout.",
    ),
):
    """Analyze infrastructure utilization and generate evidence-based rightsizing recommendations."""
    setup_logging(level="WARNING")
    settings = get_settings()
    op_start = time.monotonic()

    # 1. Parse and validate filters
    prov_filter: Optional[CloudProvider] = None
    if provider:
        p_clean = provider.strip().lower()
        if p_clean in [p.value for p in CloudProvider]:
            prov_filter = CloudProvider(p_clean)
        else:
            record_finops_operation("optimize", "failure")
            console.print(
                Panel(
                    f"[bold red]Unknown provider '{provider}'. Allowed: {[p.value for p in CloudProvider]}[/bold red]",
                    title="Invalid Argument",
                )
            )
            raise typer.Exit(code=1)

    cat_filter: Optional[ResourceCategory] = None
    if category:
        c_clean = category.strip().lower()
        if c_clean in [c.value for c in ResourceCategory]:
            cat_filter = ResourceCategory(c_clean)
        else:
            record_finops_operation("optimize", "failure")
            console.print(
                Panel(
                    f"[bold red]Unknown category '{category}'. Allowed: {[c.value for c in ResourceCategory]}[/bold red]",
                    title="Invalid Argument",
                )
            )
            raise typer.Exit(code=1)

    risk_limit: Optional[OptimizationRisk] = None
    if risk:
        r_clean = risk.strip().upper()
        if r_clean in [rk.value for rk in OptimizationRisk]:
            risk_limit = OptimizationRisk(r_clean)
        else:
            record_finops_operation("optimize", "failure")
            console.print(
                Panel(
                    f"[bold red]Unknown risk ceiling '{risk}'. Allowed: {[rk.value for rk in OptimizationRisk]}[/bold red]",
                    title="Invalid Argument",
                )
            )
            raise typer.Exit(code=1)

    # 2. Ingest resource inventory and telemetry
    try:
        if inventory_file:
            resources = load_cloud_resources(
                path=inventory_file,
                allowed_root=allowed_root,
                max_records=settings.finops_max_records,
                max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
            )
            utilization_map = {}
            if utilization_file:
                raw_utils = load_resource_utilization(
                    path=utilization_file,
                    allowed_root=allowed_root,
                    max_records=settings.finops_max_records,
                    max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
                )
                for u in raw_utils:
                    utilization_map.setdefault(u.resource_id, []).append(u)
        else:
            resources, utilization_map = get_synthetic_resource_inventory()
    except BillingDataError as e:
        record_finops_operation("optimize", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Data Ingestion Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("optimize", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Failed to load optimization datasets: {safe_msg}[/bold red]",
                title="Import Error",
            )
        )
        raise typer.Exit(code=1)

    # 3. Execute optimization analysis
    optimizer = ResourceOptimizer(
        idle_cpu_threshold=settings.finops_idle_cpu_threshold_pct,
        underutilized_cpu_threshold=settings.finops_underutilized_cpu_threshold_pct,
        idle_memory_threshold=settings.finops_idle_memory_threshold_pct,
    )
    try:
        summary = optimizer.optimize_inventory(
            resources=resources,
            utilization_map=utilization_map,
            base_currency=settings.finops_default_currency,
            provider_filter=prov_filter,
            category_filter=cat_filter,
            region_filter=region,
            min_savings=min_savings,
            risk_limit=risk_limit,
        )
    except CurrencyMismatchError as e:
        record_finops_operation("optimize", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Currency Mismatch Error"))
        raise typer.Exit(code=1)
    except OptimizationError as e:
        record_finops_operation("optimize", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Optimization Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("optimize", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Optimization analysis failed: {safe_msg}[/bold red]",
                title="Analysis Error",
            )
        )
        raise typer.Exit(code=1)

    # 4. Save output report if requested
    if output_path:
        try:
            output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            record_finops_operation("optimize", "failure")
            safe_msg = escape(_sanitize_cli_text(str(e)))
            console.print(
                Panel(
                    f"[bold red]Failed to write output report: {safe_msg}[/bold red]",
                    title="IO Error",
                )
            )
            raise typer.Exit(code=1)

    # 5. Output format: JSON or Rich Console
    if json_output:
        op_duration = max(0.0, time.monotonic() - op_start)
        record_finops_operation("optimize", "success", duration_seconds=op_duration)
        typer.echo(summary.model_dump_json(indent=2))
        return

    # Rich human-readable display
    def _fmt(amount: float) -> str:
        return f"${amount:,.2f}"

    console.print(
        Panel(
            f"[bold]Total Potential Monthly Savings:[/bold] [bold green]{_fmt(summary.total_monthly_savings)} {summary.currency}[/bold green]\n"
            f"[bold]Total Potential Annual Savings:[/bold] [bold green]{_fmt(summary.total_annual_savings)} {summary.currency}[/bold green]\n"
            f"  - [cyan]Guaranteed Savings (Orphan Asset Cleanup):[/cyan] {_fmt(summary.guaranteed_monthly_savings)}/mo (Annual: {_fmt(summary.guaranteed_annual_savings)})\n"
            f"  - [yellow]Estimated Savings (Rightsizing & Unverified Commitments):[/yellow] {_fmt(summary.estimated_monthly_savings)}/mo (Annual: {_fmt(summary.estimated_annual_savings)})\n"
            f"[bold]Recommendations Generated:[/bold] {summary.recommendations_count}\n"
            f"[bold]Breakdown by Action:[/bold] {dict(summary.recommendations_by_action)}\n"
            f"[bold]Breakdown by Risk:[/bold] {dict(summary.recommendations_by_risk)}",
            title="[bold cyan]Cloud FinOps Rightsizing & Optimization Report[/bold cyan]",
        )
    )

    if summary.recommendations:
        rec_table = Table(title="Actionable Optimization Recommendations", show_header=True)
        rec_table.add_column("Resource Identifier", style="bold cyan")
        rec_table.add_column("Service", style="green")
        rec_table.add_column("Action", style="bold yellow")
        rec_table.add_column("Risk", justify="center")
        rec_table.add_column("Monthly Savings", justify="right", style="bold green")
        rec_table.add_column("Annual Savings", justify="right")
        rec_table.add_column("Confidence", justify="right")
        rec_table.add_column("Top Evidence", style="dim")

        for r in summary.recommendations:
            risk_style = (
                "green"
                if r.risk == OptimizationRisk.LOW
                else ("yellow" if r.risk == OptimizationRisk.MEDIUM else "red")
            )
            top_ev = r.evidence[0] if r.evidence else "N/A"
            rec_table.add_row(
                escape(r.resource_id),
                escape(r.service_name),
                escape(r.action.value),
                f"[{risk_style}]{escape(r.risk.value)}[/{risk_style}]",
                f"{_fmt(r.estimated_monthly_savings)}",
                f"{_fmt(r.estimated_annual_savings)}",
                f"{r.confidence_score:.0%}",
                escape(top_ev),
            )
        console.print(rec_table)
    else:
        console.print(
            "[dim]No resources met optimization criteria (all workloads healthy or within headroom bounds).[/dim]"
        )

    console.print(
        "\n[dim yellow]Advisory Notice: Recommendations are strictly advisory and require human engineering review prior to execution. Automated modification of cloud infrastructure is disabled.[/dim yellow]"
    )
    op_duration = max(0.0, time.monotonic() - op_start)
    record_finops_operation("optimize", "success", duration_seconds=op_duration)


@finops_app.command("anomalies")
def detect_cost_anomalies(
    billing_file: Optional[Path] = typer.Option(
        None,
        "--billing-file",
        "--file",
        "-f",
        help="Path to JSON or CSV cloud billing records file (defaults to built-in synthetic benchmark).",
    ),
    service: Optional[str] = typer.Option(
        None,
        "--service",
        "-s",
        help="Filter anomalies for a specific cloud service (e.g. 'AmazonEC2', 'CloudSQL').",
    ),
    account: Optional[str] = typer.Option(
        None,
        "--account",
        "-a",
        help="Filter anomalies for a specific account or subscription ID.",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="Filter anomalies for a specific cloud region (e.g. 'us-east-1').",
    ),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        help="Filter anomalies by minimum severity ('low', 'medium', 'high', 'critical').",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Filter records starting from this date (format: YYYY-MM-DD).",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="Filter records up to this date (format: YYYY-MM-DD).",
    ),
    z_threshold: Optional[float] = typer.Option(
        None,
        "--z-threshold",
        "-z",
        help="Z-score threshold for anomaly sensitivity (overrides configuration default).",
    ),
    allowed_root: Optional[Path] = typer.Option(
        None,
        "--allowed-root",
        help="Permitted filesystem root directory to enforce path traversal safety.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save full JSON anomaly report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw anomaly report as JSON to stdout.",
    ),
):
    """Detect statistical cost anomalies and expenditure surges across multi-dimensional cloud billing data."""
    setup_logging(level="WARNING")
    settings = get_settings()
    op_start = time.monotonic()

    # 1. Parse date filters
    parsed_start_date: Optional[datetime.date] = None
    if start_date:
        try:
            parsed_start_date = datetime.date.fromisoformat(start_date.strip())
        except ValueError:
            record_finops_operation("anomalies", "failure")
            console.print(
                Panel(
                    f"[bold red]Invalid --start-date format '{start_date}'. Expected YYYY-MM-DD.[/bold red]",
                    title="Invalid Date Format",
                )
            )
            raise typer.Exit(code=1)

    parsed_end_date: Optional[datetime.date] = None
    if end_date:
        try:
            parsed_end_date = datetime.date.fromisoformat(end_date.strip())
        except ValueError:
            record_finops_operation("anomalies", "failure")
            console.print(
                Panel(
                    f"[bold red]Invalid --end-date format '{end_date}'. Expected YYYY-MM-DD.[/bold red]",
                    title="Invalid Date Format",
                )
            )
            raise typer.Exit(code=1)

    if parsed_start_date and parsed_end_date and parsed_start_date > parsed_end_date:
        record_finops_operation("anomalies", "failure")
        console.print(
            Panel(
                f"[bold red]--start-date ({parsed_start_date}) cannot be later than --end-date ({parsed_end_date}).[/bold red]",
                title="Invalid Date Range",
            )
        )
        raise typer.Exit(code=1)

    # 2. Parse severity filter
    sev_filter: Optional[AnomalySeverity] = None
    if severity:
        try:
            sev_filter = AnomalySeverity(severity.strip().upper())
        except ValueError:
            record_finops_operation("anomalies", "failure")
            valid_sevs = ", ".join(s.value.lower() for s in AnomalySeverity)
            console.print(
                Panel(
                    f"[bold red]Invalid --severity '{severity}'. Valid options: {valid_sevs}[/bold red]",
                    title="Invalid Severity",
                )
            )
            raise typer.Exit(code=1)

    # 3. Load billing records
    try:
        if billing_file:
            records, _ = load_billing_records(
                path=billing_file,
                allowed_root=allowed_root,
                max_records=settings.finops_max_records,
                max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
            )
        else:
            records = get_synthetic_anomaly_and_forecast_dataset()
    except BillingDataError as e:
        record_finops_operation("anomalies", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Billing Data Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("anomalies", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Failed to load billing records: {safe_msg}[/bold red]",
                title="Import Error",
            )
        )
        raise typer.Exit(code=1)

    # 4. Execute anomaly detection
    threshold = (
        z_threshold if z_threshold is not None else settings.finops_anomaly_z_score_threshold
    )
    detector = CostAnomalyDetector(
        z_score_threshold=threshold,
        min_observations=settings.finops_anomaly_min_observations,
        rolling_window_days=settings.finops_anomaly_rolling_window_days,
    )

    try:
        summary = detector.detect_anomalies(
            records=records,
            base_currency=settings.finops_default_currency,
            service_filter=service,
            account_filter=account,
            region_filter=region,
            severity_filter=sev_filter,
            start_date=parsed_start_date,
            end_date=parsed_end_date,
        )
    except CurrencyMismatchError as e:
        record_finops_operation("anomalies", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Currency Mismatch Error"))
        raise typer.Exit(code=1)
    except CostAnomalyError as e:
        record_finops_operation("anomalies", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Cost Anomaly Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("anomalies", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Anomaly detection failed: {safe_msg}[/bold red]",
                title="Detection Error",
            )
        )
        raise typer.Exit(code=1)

    # 5. Save output file if requested
    if output_path:
        try:
            output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            record_finops_operation("anomalies", "failure")
            safe_msg = escape(_sanitize_cli_text(str(e)))
            console.print(
                Panel(
                    f"[bold red]Failed to write output file: {safe_msg}[/bold red]",
                    title="IO Error",
                )
            )
            raise typer.Exit(code=1)

    # 6. JSON output
    if json_output:
        op_duration = max(0.0, time.monotonic() - op_start)
        record_finops_operation("anomalies", "success", duration_seconds=op_duration)
        typer.echo(summary.model_dump_json(indent=2))
        return

    # 7. Rich Human-readable display
    def _fmt_money(val: float) -> str:
        return f"${val:,.2f}"

    start_str = summary.period_start.isoformat() if summary.period_start else "N/A"
    end_str = summary.period_end.isoformat() if summary.period_end else "N/A"

    console.print(
        Panel(
            f"[bold]Total Anomalies Detected:[/bold] [bold red]{summary.total_anomalies}[/bold red]\n"
            f"[bold]Total Excess Spend Above Baseline:[/bold] [bold yellow]{_fmt_money(summary.total_excess_spend)} {summary.currency}[/bold yellow]\n"
            f"[bold]Breakdown by Severity:[/bold] {dict(summary.anomalies_by_severity)}\n"
            f"[bold]Breakdown by Service:[/bold] {dict(summary.anomalies_by_service)}\n"
            f"[bold]Analysis Time Window:[/bold] {start_str} -> {end_str}",
            title="[bold red]Cloud FinOps Cost Anomaly Detection Report[/bold red]",
        )
    )

    if summary.anomalies:
        table = Table(title="Detected Statistical Spending Surges", show_header=True)
        table.add_column("Date", style="cyan")
        table.add_column("Dimension / Service", style="bold yellow")
        table.add_column("Expected", justify="right")
        table.add_column("Actual", justify="right", style="bold")
        table.add_column("Delta ($ / %)", justify="right")
        table.add_column("Z-Score", justify="right")
        table.add_column("Severity", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Top Evidence", style="dim")

        for anom in summary.anomalies:
            sev_style = (
                "bold red"
                if anom.severity == AnomalySeverity.CRITICAL
                else (
                    "red"
                    if anom.severity == AnomalySeverity.HIGH
                    else ("yellow" if anom.severity == AnomalySeverity.MEDIUM else "blue")
                )
            )
            dim_label = anom.service_name or (
                f"Account: {anom.account_id}" if anom.account_id else "Total Spend"
            )
            delta_str = f"+{_fmt_money(anom.cost_delta)} (+{anom.percentage_increase:.0f}%)"
            top_ev = anom.evidence[0] if anom.evidence else anom.explanation

            table.add_row(
                anom.detected_date.isoformat(),
                escape(dim_label),
                _fmt_money(anom.expected_cost),
                _fmt_money(anom.actual_cost),
                delta_str,
                f"{anom.z_score:.2f}σ",
                f"[{sev_style}]{anom.severity.value}[/{sev_style}]",
                f"{anom.confidence_score:.0%}",
                escape(top_ev),
            )
        console.print(table)
    else:
        console.print("[dim]No spending anomalies detected within specified parameters.[/dim]")

    console.print(
        "\n[dim yellow]Advisory Notice: Anomalies represent statistical deviations in historical spend and require engineering investigation. Automated remediation or resource termination is disabled.[/dim yellow]"
    )
    op_duration = max(0.0, time.monotonic() - op_start)
    record_finops_operation("anomalies", "success", duration_seconds=op_duration)


@finops_app.command("forecast")
def forecast_cloud_expenditure(
    billing_file: Optional[Path] = typer.Option(
        None,
        "--billing-file",
        "--file",
        "-f",
        help="Path to JSON or CSV cloud billing records file (defaults to built-in synthetic benchmark).",
    ),
    horizon: Optional[int] = typer.Option(
        None,
        "--horizon",
        "-h",
        help="Forecast horizon in days (defaults to configuration default).",
    ),
    frequency: str = typer.Option(
        "daily",
        "--frequency",
        help="Forecast granularity: 'daily' or 'monthly' (default: 'daily').",
    ),
    method: str = typer.Option(
        "linear_regression",
        "--method",
        "-m",
        help="Forecasting algorithm: 'linear_regression', 'weighted_moving_average', 'exponential_smoothing', or 'historical_average'.",
    ),
    service: Optional[str] = typer.Option(
        None,
        "--service",
        "-s",
        help="Filter forecast to a specific cloud service.",
    ),
    account: Optional[str] = typer.Option(
        None,
        "--account",
        "-a",
        help="Filter forecast to a specific account or subscription ID.",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="Filter forecast to a specific cloud region.",
    ),
    confidence: Optional[float] = typer.Option(
        None,
        "--confidence",
        help="Prediction interval confidence level (0.50 - 0.99, defaults to config default).",
    ),
    allowed_root: Optional[Path] = typer.Option(
        None,
        "--allowed-root",
        help="Permitted filesystem root directory to enforce path traversal safety.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save full JSON forecast report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw forecast summary report as JSON to stdout.",
    ),
):
    """Generate forward-looking expenditure projections with prediction intervals (P10/P50/P90)."""
    setup_logging(level="WARNING")
    settings = get_settings()
    op_start = time.monotonic()

    # 1. Validate inputs
    norm_freq = frequency.strip().lower()
    if norm_freq not in ["daily", "monthly"]:
        record_finops_operation("forecast", "failure")
        console.print(
            Panel(
                f"[bold red]Invalid --frequency '{frequency}'. Must be 'daily' or 'monthly'.[/bold red]",
                title="Invalid Frequency",
            )
        )
        raise typer.Exit(code=1)

    target_method: ForecastMethod
    try:
        target_method = ForecastMethod(method.strip().lower())
    except ValueError:
        record_finops_operation("forecast", "failure")
        valid_methods = ", ".join(m.value for m in ForecastMethod)
        console.print(
            Panel(
                f"[bold red]Invalid --method '{method}'. Valid options: {valid_methods}[/bold red]",
                title="Invalid Method",
            )
        )
        raise typer.Exit(code=1)

    if horizon is not None and horizon < 1:
        record_finops_operation("forecast", "failure")
        console.print(
            Panel(
                f"[bold red]Invalid --horizon '{horizon}'. Must be at least 1 day.[/bold red]",
                title="Invalid Horizon",
            )
        )
        raise typer.Exit(code=1)

    if confidence is not None and not (0.50 <= confidence <= 0.99):
        record_finops_operation("forecast", "failure")
        console.print(
            Panel(
                f"[bold red]Invalid --confidence '{confidence}'. Must be between 0.50 and 0.99.[/bold red]",
                title="Invalid Confidence",
            )
        )
        raise typer.Exit(code=1)

    # 2. Load billing records
    try:
        if billing_file:
            records, _ = load_billing_records(
                path=billing_file,
                allowed_root=allowed_root,
                max_records=settings.finops_max_records,
                max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
            )
        else:
            records = get_synthetic_anomaly_and_forecast_dataset()
    except BillingDataError as e:
        record_finops_operation("forecast", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Billing Data Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("forecast", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Failed to load billing records: {safe_msg}[/bold red]",
                title="Import Error",
            )
        )
        raise typer.Exit(code=1)

    # 3. Execute forecast
    h_days = horizon if horizon is not None else settings.finops_forecast_default_horizon_days
    conf_level = confidence if confidence is not None else settings.finops_forecast_confidence_level
    forecaster = CostForecaster(
        default_horizon_days=h_days,
        confidence_level=conf_level,
        min_observations=settings.finops_forecast_min_observations,
    )

    try:
        cost_forecast = forecaster.forecast(
            records=records,
            horizon_days=h_days,
            method=target_method,
            base_currency=settings.finops_default_currency,
            service_filter=service,
            account_filter=account,
            region_filter=region,
            frequency=norm_freq,
        )
        summary = forecaster.summarize_forecast(cost_forecast)
    except CurrencyMismatchError as e:
        record_finops_operation("forecast", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Currency Mismatch Error"))
        raise typer.Exit(code=1)
    except InsufficientDataError as e:
        record_finops_operation("forecast", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Insufficient Data Error"))
        raise typer.Exit(code=1)
    except CostForecastError as e:
        record_finops_operation("forecast", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Cost Forecast Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("forecast", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(f"[bold red]Forecasting failed: {safe_msg}[/bold red]", title="Forecast Error")
        )
        raise typer.Exit(code=1)

    # 4. Save output file if requested
    if output_path:
        try:
            output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            record_finops_operation("forecast", "failure")
            safe_msg = escape(_sanitize_cli_text(str(e)))
            console.print(
                Panel(
                    f"[bold red]Failed to write output file: {safe_msg}[/bold red]",
                    title="IO Error",
                )
            )
            raise typer.Exit(code=1)

    # 5. JSON output
    if json_output:
        op_duration = max(0.0, time.monotonic() - op_start)
        record_finops_operation("forecast", "success", duration_seconds=op_duration)
        typer.echo(summary.model_dump_json(indent=2))
        return

    # 6. Rich human-readable display
    def _fmt_money(val: float) -> str:
        return f"${val:,.2f}"

    trend_styles = {
        "increasing": "bold red",
        "decreasing": "bold green",
        "stable": "bold cyan",
    }
    t_style = trend_styles.get(summary.trend, "bold yellow")

    panel_text = (
        f"[bold]Forecast Window:[/bold] {summary.forecast_start_date} -> {summary.forecast_end_date} ({summary.horizon_days} {'months' if norm_freq == 'monthly' else 'days'})\n"
        f"[bold]Algorithm / Method:[/bold] {summary.method}\n"
        f"[bold]Trajectory / Trend:[/bold] [{t_style}]{summary.trend.upper()}[/{t_style}]\n"
        f"[bold]Confidence Level:[/bold] {summary.confidence_level:.0%}\n"
        f"[bold]Historical Days Analyzed:[/bold] {summary.historical_days_analyzed}\n"
        f"--------------------------------------------------\n"
        f"[bold]Projected Spend (P50 Median):[/bold] [bold cyan]{_fmt_money(summary.total_projected_cost)} {summary.currency}[/bold cyan]\n"
        f"[bold]Optimistic Spend (P10 Bound):[/bold] [bold green]{_fmt_money(summary.total_p10_cost)} {summary.currency}[/bold green]\n"
        f"[bold]Conservative Spend (P90 Bound):[/bold] [bold red]{_fmt_money(summary.total_p90_cost)} {summary.currency}[/bold red]\n"
        f"[bold]Projected {'Monthly' if norm_freq == 'monthly' else 'Daily'} Average:[/bold] {_fmt_money(summary.daily_average_cost)} {summary.currency}"
    )

    if cost_forecast.evaluation_metrics:
        mae = cost_forecast.evaluation_metrics.get("mae", 0.0)
        rmse = cost_forecast.evaluation_metrics.get("rmse", 0.0)
        mape = cost_forecast.evaluation_metrics.get("mape", 0.0)
        panel_text += (
            f"\n--------------------------------------------------\n"
            f"[bold]In-Sample Goodness of Fit:[/bold] MAE={_fmt_money(mae)}, RMSE={_fmt_money(rmse)}, MAPE={mape:.1f}%"
        )

    console.print(
        Panel(
            panel_text,
            title="[bold cyan]Cloud FinOps Expenditure Forecast Report[/bold cyan]",
        )
    )

    if cost_forecast.points:
        table = Table(title=f"Projected Spend Trajectory ({norm_freq.title()})", show_header=True)
        table.add_column("Date", style="cyan")
        table.add_column("Optimistic (P10)", justify="right", style="green")
        table.add_column("Projected (P50)", justify="right", style="bold cyan")
        table.add_column("Conservative (P90)", justify="right", style="red")
        table.add_column("Uncertainty Range", justify="right", style="dim")

        points_to_display = cost_forecast.points
        if len(points_to_display) > 31:
            points_to_display = points_to_display[:15] + points_to_display[-5:]

        for idx, pt in enumerate(points_to_display):
            if len(cost_forecast.points) > 31 and idx == 15:
                table.add_row("...", "...", "...", "...", "...")
            spread = pt.upper_bound - pt.lower_bound
            table.add_row(
                pt.date.isoformat(),
                _fmt_money(pt.lower_bound),
                _fmt_money(pt.projected_cost),
                _fmt_money(pt.upper_bound),
                f"±{_fmt_money(spread / 2.0)}",
            )
        console.print(table)

    console.print(
        "\n[dim yellow]Advisory Notice: Projections are statistical forecasts based on historical spending trajectories. In-sample fit metrics indicate historical model alignment and do not guarantee future predictive accuracy. Zero-clamped prediction intervals are Gaussian approximations for planning purposes and are not financial guarantees.[/dim yellow]"
    )
    op_duration = max(0.0, time.monotonic() - op_start)
    record_finops_operation("forecast", "success", duration_seconds=op_duration)


@finops_app.command("advise")
def advise_finops_strategy(
    billing_file: Optional[Path] = typer.Option(
        None,
        "--billing-file",
        "--file",
        "-f",
        help="Path to JSON or CSV cloud billing records file (defaults to built-in synthetic benchmark).",
    ),
    inventory_file: Optional[Path] = typer.Option(
        None,
        "--inventory-file",
        "-i",
        help="Path to JSON cloud resource inventory file (defaults to built-in synthetic inventory).",
    ),
    utilization_file: Optional[Path] = typer.Option(
        None,
        "--utilization-file",
        "-u",
        help="Optional path to JSON resource utilization metrics dataset.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider for FinOps advisory synthesis ('ollama', 'gemini', or custom).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model identifier to utilize for advisory synthesis.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Execute 100% offline rule-based deterministic synthesis without calling external LLMs.",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Optional guidance query or focal strategic intent (e.g. 'Focus on storage cleanup').",
    ),
    allowed_root: Optional[Path] = typer.Option(
        None,
        "--allowed-root",
        help="Permitted filesystem root directory to enforce path traversal safety.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save full JSON advisory report.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw FinOps advisory report as JSON to stdout.",
    ),
):
    """Synthesize holistic AI-assisted FinOps advisory recommendations and strategic cost optimization."""
    setup_logging(level="WARNING")
    settings = get_settings()
    op_start = time.monotonic()

    # 1. Ingest billing telemetry
    try:
        if billing_file:
            records, _ = load_billing_records(
                path=billing_file,
                allowed_root=allowed_root,
                max_records=settings.finops_max_records,
                max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
            )
        else:
            records = get_synthetic_anomaly_and_forecast_dataset()
    except BillingDataError as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Billing Data Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Failed to load billing records: {safe_msg}[/bold red]",
                title="Import Error",
            )
        )
        raise typer.Exit(code=1)

    # 2. Ingest resource inventory and utilization
    try:
        if inventory_file:
            resources = load_cloud_resources(
                path=inventory_file,
                allowed_root=allowed_root,
                max_records=settings.finops_max_records,
                max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
            )
            utilization_map = {}
            if utilization_file:
                raw_utils = load_resource_utilization(
                    path=utilization_file,
                    allowed_root=allowed_root,
                    max_records=settings.finops_max_records,
                    max_file_size_bytes=settings.finops_max_billing_file_size_bytes,
                )
                for u in raw_utils:
                    utilization_map.setdefault(u.resource_id, []).append(u)
        else:
            resources, utilization_map = get_synthetic_resource_inventory()
    except BillingDataError as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Data Ingestion Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Failed to load optimization datasets: {safe_msg}[/bold red]",
                title="Import Error",
            )
        )
        raise typer.Exit(code=1)

    # 3. Execute deterministic FinOps analytics engines
    try:
        analyzer = CostAnalyzer()
        cost_summary = analyzer.summarize(
            records=records, base_currency=settings.finops_default_currency
        )

        optimizer = ResourceOptimizer(
            idle_cpu_threshold=settings.finops_idle_cpu_threshold_pct,
            underutilized_cpu_threshold=settings.finops_underutilized_cpu_threshold_pct,
            idle_memory_threshold=settings.finops_idle_memory_threshold_pct,
        )
        opt_summary = optimizer.optimize_inventory(
            resources=resources,
            utilization_map=utilization_map,
            base_currency=settings.finops_default_currency,
        )

        detector = CostAnomalyDetector(
            z_score_threshold=settings.finops_anomaly_z_score_threshold,
            min_observations=settings.finops_anomaly_min_observations,
            rolling_window_days=settings.finops_anomaly_rolling_window_days,
        )
        anomaly_summary = detector.detect_anomalies(
            records=records,
            base_currency=settings.finops_default_currency,
        )

        forecaster = CostForecaster(
            default_horizon_days=settings.finops_forecast_default_horizon_days,
            confidence_level=settings.finops_forecast_confidence_level,
            min_observations=settings.finops_forecast_min_observations,
        )
        cost_forecast = forecaster.forecast(
            records=records,
            horizon_days=settings.finops_forecast_default_horizon_days,
            base_currency=settings.finops_default_currency,
        )
        forecast_summary = forecaster.summarize_forecast(cost_forecast)
    except CurrencyMismatchError as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Currency Mismatch Error"))
        raise typer.Exit(code=1)
    except (OptimizationError, CostAnomalyError, CostForecastError, InsufficientDataError) as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="FinOps Engine Error"))
        raise typer.Exit(code=1)
    except Exception as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]Analytics execution failed: {safe_msg}[/bold red]", title="Engine Error"
            )
        )
        raise typer.Exit(code=1)

    # 4. Execute AI FinOps Advisory Agent
    try:
        agent = FinOpsAgent(
            provider_name=provider,
            model=model,
            offline=offline,
            settings=settings,
        )
        agent.initialize()
        report = agent.advise(
            cost_summary=cost_summary,
            optimization_summary=opt_summary,
            anomaly_summary=anomaly_summary,
            forecast_summary=forecast_summary,
            user_query=query,
        )
        op_duration = time.monotonic() - op_start
        record_finops_operation("advise", "success", duration_seconds=op_duration)
    except Exception as e:
        record_finops_operation("advise", "failure")
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(
            Panel(
                f"[bold red]FinOps advisory agent failed: {safe_msg}[/bold red]",
                title="Advisory Failed",
            )
        )
        raise typer.Exit(code=1)

    # 5. Output file export
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            console.print(f"[green]Successfully saved advisory report to {output_path}[/green]")
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Failed to write advisory report to '{output_path}': {escape(str(e))}[/bold red]",
                    title="Export Error",
                )
            )
            raise typer.Exit(code=1)

    # 6. JSON output
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return

    # 7. Rich Human-Readable Output
    def _fmt_money(val: float) -> str:
        return f"{report.currency} {val:,.2f}" if report.currency != "USD" else f"${val:,.2f}"

    hist_excess_note = ""
    if report.historical_excess_spend > 0:
        hist_excess_note = f" | [bold]Historical Anomaly Variance:[/bold] [yellow]{_fmt_money(report.historical_excess_spend)}[/yellow] (Cost Avoidance)"

    panel_text = (
        f"[bold]Executive Summary:[/bold]\n{report.executive_summary}\n\n"
        f"--------------------------------------------------\n"
        f"[bold]Total Identified Monthly Savings:[/bold] [bold green]{_fmt_money(report.total_monthly_savings)} {report.currency}[/bold green] "
        f"([green]Guaranteed: {_fmt_money(report.guaranteed_monthly_savings)}[/green] | [cyan]Estimated: {_fmt_money(report.estimated_monthly_savings)}[/cyan])\n"
        f"[bold]Projected Annual Savings:[/bold] [bold green]{_fmt_money(report.total_annual_savings)} {report.currency}[/bold green]\n"
        f"[bold]Anomalies Accounted:[/bold] {report.anomalies_detected}{hist_excess_note} | [bold]Forecast Trajectory:[/bold] {report.forecast_trend.upper()}\n"
        f"[bold]Advisory Provider / Engine:[/bold] {report.provider.upper()}"
        + (f" ({report.model})" if report.model else "")
    )

    console.print(
        Panel(
            panel_text,
            title="[bold green]AI-Assisted FinOps Strategic Advisory Report[/bold green]",
        )
    )

    if report.recommendations:
        rec_table = Table(title="Prioritized Cost Optimization Recommendations", show_header=True)
        rec_table.add_column("Priority", style="bold", width=10)
        rec_table.add_column("Category", style="cyan", width=12)
        rec_table.add_column("Type", width=12)
        rec_table.add_column("Monthly Savings", justify="right", style="green", width=16)
        rec_table.add_column("Confidence", justify="right", width=12)
        rec_table.add_column("Recommendation & Action")

        prio_colors = {
            "critical": "bold red",
            "high": "red",
            "medium": "yellow",
            "low": "cyan",
        }

        for rec in report.recommendations:
            p_style = prio_colors.get(rec.priority.lower(), "white")
            type_style = "green" if rec.savings_type == "guaranteed" else "cyan"
            rec_table.add_row(
                f"[{p_style}]{rec.priority.upper()}[/{p_style}]",
                rec.category.upper(),
                f"[{type_style}]{rec.savings_type.capitalize()}[/{type_style}]",
                _fmt_money(rec.potential_monthly_savings),
                f"{rec.confidence_score:.0%}",
                f"[bold]{rec.title}[/bold]\n[dim]{rec.action}[/dim]",
            )
        console.print(rec_table)

    console.print(f"\n[dim yellow]Advisory Disclaimer: {report.disclaimer}[/dim yellow]")


def main():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    main()
