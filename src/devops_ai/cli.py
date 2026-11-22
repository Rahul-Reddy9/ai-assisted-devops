# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Command-line interface entry point for DevOps AI."""

from pathlib import Path
import re
from typing import Optional
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from devops_ai import __version__
from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.core.logging import setup_logging
from devops_ai.generators.dockerfile import DockerfileGenerator
from devops_ai.generators.k8s_manifest import K8sManifestGenerator
from devops_ai.llm.router import LLMRouter

app = typer.Typer(
    name="devops-ai",
    help="Unified AI-Assisted DevOps suite for infrastructure generation, observability, and incident diagnosis.",
    add_completion=False,
)
generate_app = typer.Typer(name="generate", help="Generate infrastructure manifests (Dockerfiles, Kubernetes YAML).")
provider_app = typer.Typer(name="provider", help="Inspect and test LLM provider connectivity.")
app.add_typer(generate_app)
app.add_typer(provider_app)

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
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider name (ollama or gemini)."),
    host: Optional[str] = typer.Option(None, "--host", help="Ollama host URL override."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override."),
    timeout: Optional[float] = typer.Option(None, "--timeout", "-t", help="Timeout in seconds."),
    verify_remote: bool = typer.Option(False, "--verify-remote", help="Perform live remote verification ping to LLM endpoint."),
):
    """Check connectivity and health of the configured LLM provider."""
    try:
        if provider and provider.lower() not in ("ollama", "gemini"):
            console.print(Panel(
                f"[bold red]Unsupported provider: '{escape(provider)}'. Must be 'ollama' or 'gemini'.[/bold red]",
                title="Input Error",
            ))
            raise typer.Exit(code=2)

        if timeout is not None and timeout <= 0:
            console.print(Panel("[bold red]Timeout must be a positive number of seconds.[/bold red]", title="Input Error"))
            raise typer.Exit(code=2)

        llm = LLMRouter.get_provider(provider_name=provider, model=model, host=host, timeout=timeout)
        is_healthy = llm.check_health(verify_remote=verify_remote)

        if llm.provider_name == "gemini":
            if not is_healthy:
                console.print(Panel(
                    "[bold red]Gemini check failed.[/bold red]\n"
                    "Ensure GOOGLE_API_KEY is configured and valid.",
                    title="Provider Status",
                ))
                raise typer.Exit(code=1)
            elif not verify_remote:
                console.print(Panel(
                    f"[bold green]Gemini credentials configured.[/bold green]\n"
                    f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                    f"[yellow]Local configuration verified; remote connectivity not tested.[/yellow]\n"
                    f"Pass [cyan]--verify-remote[/cyan] to test live API connectivity.",
                    title="Provider Status",
                ))
            else:
                console.print(Panel(
                    f"[bold green]Gemini is healthy and connected.[/bold green]\n"
                    f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                    f"Live remote verification succeeded.",
                    title="Provider Status",
                ))
        else:
            endpoint = llm.host or "unknown"
            if is_healthy:
                if verify_remote:
                    from devops_ai.llm.ollama_provider import OllamaProvider
                    installed = []
                    if isinstance(llm, OllamaProvider):
                        installed = llm.get_installed_models()
                    model_status = f" (model '{escape(llm.model_name)}' installed)" if llm.model_name in installed else ""
                    console.print(Panel(
                        f"[bold green]Provider 'ollama' is healthy and reachable.[/bold green]{model_status}\n"
                        f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                        f"Endpoint: [cyan]{escape(endpoint)}[/cyan]",
                        title="Provider Status",
                    ))
                else:
                    console.print(Panel(
                        f"[bold green]Ollama configuration valid.[/bold green]\n"
                        f"Model: [cyan]{escape(llm.model_name)}[/cyan]\n"
                        f"Endpoint: [cyan]{escape(endpoint)}[/cyan]\n"
                        f"[yellow]Local configuration verified; pass --verify-remote to ping daemon.[/yellow]",
                        title="Provider Status",
                    ))
            else:
                console.print(Panel(
                    f"[bold red]Provider 'ollama' is unreachable at {escape(endpoint)}.[/bold red]\n"
                    f"Ensure Ollama daemon is running ('ollama serve').",
                    title="Provider Status",
                ))
                raise typer.Exit(code=1)
    except DevOpsAIError as e:
        safe_msg = escape(_sanitize_cli_text(e.message))
        console.print(Panel(f"[bold red]{safe_msg}[/bold red]", title="Configuration / Provider Error"))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        safe_msg = escape(_sanitize_cli_text(str(e)))
        console.print(Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error"))
        raise typer.Exit(code=1)


@generate_app.command("dockerfile")
def generate_dockerfile_cmd(
    language: str = typer.Argument(..., help="Programming language or stack (e.g. python, nodejs, golang)."),
    port: int = typer.Option(8000, "--port", "-p", help="Target container application port."),
    base_image: Optional[str] = typer.Option(None, "--base-image", "-b", help="Base container image."),
    instructions: Optional[str] = typer.Option(None, "--instructions", help="Custom generation instructions."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File path to save the generated Dockerfile."),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite of output file if it already exists."),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider (ollama or gemini)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override."),
    host: Optional[str] = typer.Option(None, "--host", help="Ollama host URL override."),
    timeout: Optional[float] = typer.Option(None, "--timeout", "-t", help="Timeout in seconds."),
    fallback: Optional[bool] = typer.Option(None, "--fallback/--no-fallback", help="Override automatic fallback."),
):
    """Generate a validated, secure Dockerfile."""
    try:
        if provider and provider.lower() not in ("ollama", "gemini"):
            console.print(Panel(
                f"[bold red]Unsupported provider: '{escape(provider)}'. Must be 'ollama' or 'gemini'.[/bold red]",
                title="Input Error",
            ))
            raise typer.Exit(code=2)

        if port < 1 or port > 65535:
            console.print(Panel(f"[bold red]Port must be between 1 and 65535, got {port}.[/bold red]", title="Input Error"))
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
        console.print(Panel(escape(result.content), title=f"Generated Dockerfile ({escape(language)})", subtitle=subtitle))
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
        console.print(Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error"))
        raise typer.Exit(code=1)


@generate_app.command("k8s")
def generate_k8s_cmd(
    app_name: str = typer.Argument(..., help="Application name."),
    image: str = typer.Option(..., "--image", "-i", help="Container image (e.g. nginx:1.25, myapp:1.0)."),
    port: int = typer.Option(8080, "--port", "-p", help="Container application port."),
    replicas: int = typer.Option(1, "--replicas", "-r", help="Number of deployment replicas."),
    service_type: str = typer.Option("ClusterIP", "--service-type", "-s", help="Service type: ClusterIP, NodePort, LoadBalancer."),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Target Kubernetes namespace."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File path to save the generated manifest."),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite of output file if it already exists."),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider (ollama or gemini)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override."),
    host: Optional[str] = typer.Option(None, "--host", help="Ollama host URL override."),
    timeout: Optional[float] = typer.Option(None, "--timeout", "-t", help="Timeout in seconds."),
    fallback: Optional[bool] = typer.Option(None, "--fallback/--no-fallback", help="Override automatic fallback."),
):
    """Generate Kubernetes Deployment and Service manifests."""
    try:
        if provider and provider.lower() not in ("ollama", "gemini"):
            console.print(Panel(
                f"[bold red]Unsupported provider: '{escape(provider)}'. Must be 'ollama' or 'gemini'.[/bold red]",
                title="Input Error",
            ))
            raise typer.Exit(code=2)

        if port < 1 or port > 65535:
            console.print(Panel(f"[bold red]Port must be between 1 and 65535, got {port}.[/bold red]", title="Input Error"))
            raise typer.Exit(code=2)

        if replicas < 1:
            console.print(Panel(f"[bold red]Replicas must be an integer >= 1, got {replicas}.[/bold red]", title="Input Error"))
            raise typer.Exit(code=2)

        if service_type not in ("ClusterIP", "NodePort", "LoadBalancer"):
            console.print(Panel(
                f"[bold red]Unsupported service type: '{escape(service_type)}'. Must be ClusterIP, NodePort, or LoadBalancer.[/bold red]",
                title="Input Error",
            ))
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
        console.print(Panel(escape(result.content), title=f"Generated Kubernetes Manifests ({escape(app_name)})", subtitle=subtitle))
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
        console.print(Panel(f"[bold red]Unexpected error: {safe_msg}[/bold red]", title="Fatal Error"))
        raise typer.Exit(code=1)


def main():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    main()
