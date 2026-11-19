# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Command-line interface entry point for DevOps AI."""

from typing import Optional
import typer
from rich.console import Console
from devops_ai import __version__

app = typer.Typer(
    name="devops-ai",
    help="Unified AI-Assisted DevOps suite for infrastructure generation, observability, and incident diagnosis.",
    add_completion=False,
)
console = Console()


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
    pass


@app.command()
def version():
    """Print the DevOps AI version."""
    console.print(f"devops-ai {__version__}")


def main():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    main()
