# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for the CLI entry point."""

from unittest.mock import patch
from typer.testing import CliRunner
from devops_ai import __version__
from devops_ai.cli import app
from devops_ai.core.config import Settings
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.generators.dockerfile import DockerfileValidationResult
from devops_ai.generators.k8s_manifest import K8sValidationResult

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


def test_provider_check_invalid_provider_exit_code():
    result = runner.invoke(app, ["provider", "check", "--provider", "unsupported"])
    assert result.exit_code == 2
    assert "Unsupported provider" in result.stdout


def test_provider_check_invalid_timeout():
    result = runner.invoke(app, ["provider", "check", "--timeout", "-5"])
    assert result.exit_code == 2
    assert "Timeout must be a positive number" in result.stdout


def test_provider_check_gemini_local_only():
    with patch("devops_ai.cli.get_settings", return_value=Settings(google_api_key="valid-key")):
        with patch("devops_ai.llm.gemini_provider.get_settings", return_value=Settings(google_api_key="valid-key")):
            result = runner.invoke(app, ["provider", "check", "--provider", "gemini"])
            assert result.exit_code == 0
            assert "Gemini credentials configured" in result.stdout
            assert "remote connectivity not tested" in result.stdout


def test_provider_check_gemini_verify_remote_success():
    with patch("devops_ai.cli.get_settings", return_value=Settings(google_api_key="valid-key")):
        with patch("devops_ai.llm.gemini_provider.get_settings", return_value=Settings(google_api_key="valid-key")):
            with patch("devops_ai.llm.gemini_provider.GeminiProvider.check_health", return_value=True):
                result = runner.invoke(app, ["provider", "check", "--provider", "gemini", "--verify-remote"])
                assert result.exit_code == 0
                assert "Gemini is healthy and connected" in result.stdout


def test_provider_check_gemini_verify_remote_failure():
    with patch("devops_ai.cli.get_settings", return_value=Settings(google_api_key="valid-key")):
        with patch("devops_ai.llm.gemini_provider.get_settings", return_value=Settings(google_api_key="valid-key")):
            with patch("devops_ai.llm.gemini_provider.GeminiProvider.check_health", return_value=False):
                result = runner.invoke(app, ["provider", "check", "--provider", "gemini", "--verify-remote"])
                assert result.exit_code == 1
                assert "Gemini check failed" in result.stdout


def test_provider_check_ollama_reachable():
    with patch("devops_ai.llm.ollama_provider.OllamaProvider.check_health", return_value=True):
        result = runner.invoke(app, ["provider", "check", "--provider", "ollama"])
        assert result.exit_code == 0
        assert "Ollama configuration valid" in result.stdout


def test_provider_check_ollama_unreachable():
    with patch("devops_ai.llm.ollama_provider.OllamaProvider.check_health", return_value=False):
        result = runner.invoke(app, ["provider", "check", "--provider", "ollama", "--verify-remote"])
        assert result.exit_code == 1
        assert "is unreachable" in result.stdout


def test_generate_dockerfile_cli_invalid_port():
    result = runner.invoke(app, ["generate", "dockerfile", "python", "--port", "99999"])
    assert result.exit_code == 2
    assert "Port must be between 1 and 65535" in result.stdout


def test_generate_dockerfile_cli_invalid_provider():
    result = runner.invoke(app, ["generate", "dockerfile", "python", "--provider", "invalid"])
    assert result.exit_code == 2
    assert "Unsupported provider" in result.stdout


def test_generate_dockerfile_cli_success():
    mock_res = DockerfileValidationResult(
        is_valid=True,
        content="FROM python:3.12-slim\nUSER app",
        warnings=["Test warning"],
        provider="ollama",
        model="llama3.2:1b",
    )
    with patch("devops_ai.cli.DockerfileGenerator.generate", return_value=mock_res):
        result = runner.invoke(app, ["generate", "dockerfile", "python", "--port", "8000"])
        assert result.exit_code == 0
        assert "FROM python:3.12-slim" in result.stdout
        assert "Test warning" in result.stdout


def test_generate_dockerfile_cli_rich_escaping():
    # If LLM produces content containing rich-like tags [bold red] it must not break rendering
    mock_res = DockerfileValidationResult(
        is_valid=True,
        content="FROM python:3.12-slim\n# [bold red]Not a tag[/bold red]\nUSER app",
        warnings=[],
        provider="ollama",
        model="llama3.2:1b",
    )
    with patch("devops_ai.cli.DockerfileGenerator.generate", return_value=mock_res):
        result = runner.invoke(app, ["generate", "dockerfile", "python"])
        assert result.exit_code == 0
        assert "[bold red]" in result.stdout


def test_generate_k8s_cli_invalid_service_type():
    result = runner.invoke(app, ["generate", "k8s", "myapp", "--image", "myapp:1.0", "--service-type", "InvalidType"])
    assert result.exit_code == 2
    assert "Unsupported service type" in result.stdout


def test_generate_k8s_cli_invalid_replicas():
    result = runner.invoke(app, ["generate", "k8s", "myapp", "--image", "myapp:1.0", "--replicas", "0"])
    assert result.exit_code == 2
    assert "Replicas must be an integer >= 1" in result.stdout


def test_generate_k8s_cli_success():
    mock_res = K8sValidationResult(
        is_valid=True,
        content="apiVersion: apps/v1\nkind: Deployment\n---\napiVersion: v1\nkind: Service",
        documents=[{"kind": "Deployment"}, {"kind": "Service"}],
        warnings=[],
        provider="ollama",
        model="llama3.2:1b",
    )
    with patch("devops_ai.cli.K8sManifestGenerator.generate", return_value=mock_res):
        result = runner.invoke(app, ["generate", "k8s", "myapp", "--image", "myapp:1.0", "--port", "8080"])
        assert result.exit_code == 0
        assert "Generated Kubernetes Manifests" in result.stdout


def test_cli_error_sanitization():
    with patch("devops_ai.cli.DockerfileGenerator.generate", side_effect=DevOpsAIError("API error with AIzaSyFakeKey123456789012345678901234 in URL")):
        result = runner.invoke(app, ["generate", "dockerfile", "python"])
        assert result.exit_code == 1
        assert "AIzaSyFakeKey" not in result.stdout
        assert "[REDACTED_API_KEY]" in result.stdout


def test_cli_unexpected_exception_sanitization():
    with patch("devops_ai.cli.DockerfileGenerator.generate", side_effect=RuntimeError("Fatal error with key AIzaSyFakeKey123456789012345678901234")):
        result = runner.invoke(app, ["generate", "dockerfile", "python"])
        assert result.exit_code == 1
        assert "AIzaSyFakeKey" not in result.stdout
        assert "[REDACTED_API_KEY]" in result.stdout
        assert "Traceback" not in result.stdout
