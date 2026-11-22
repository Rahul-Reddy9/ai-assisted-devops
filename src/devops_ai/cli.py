# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Command-line interface entry point for DevOps AI."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import time
from typing import Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
import typer

from devops_ai import __version__
from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.core.logging import setup_logging
from devops_ai.generators.dockerfile import DockerfileGenerator
from devops_ai.generators.k8s_manifest import K8sManifestGenerator
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

app.add_typer(generate_app)
app.add_typer(provider_app)
app.add_typer(health_app)
app.add_typer(metrics_app)
app.add_typer(monitor_app)

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


def main():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    main()
