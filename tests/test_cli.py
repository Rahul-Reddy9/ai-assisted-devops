# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for the CLI entry point."""

from typer.testing import CliRunner
from devops_ai import __version__
from devops_ai.cli import app

runner = CliRunner()


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"devops-ai {__version__}" in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Unified AI-Assisted DevOps suite" in result.stdout
