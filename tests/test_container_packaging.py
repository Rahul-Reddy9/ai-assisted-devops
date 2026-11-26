# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Tests for production container packaging, Dockerfile architecture, and .dockerignore rules."""

import fnmatch
from pathlib import Path
import re
import pytest
from devops_ai.cli import app

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile_content() -> str:
    """Read and return Dockerfile content."""
    assert DOCKERFILE_PATH.is_file(), f"Dockerfile not found at {DOCKERFILE_PATH}"
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore_lines() -> list[str]:
    """Read and return active, non-comment lines from .dockerignore."""
    assert DOCKERIGNORE_PATH.is_file(), f".dockerignore not found at {DOCKERIGNORE_PATH}"
    lines = []
    for raw in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def test_dockerfile_and_dockerignore_exist():
    """Verify that Dockerfile and .dockerignore exist at the repository root."""
    assert DOCKERFILE_PATH.is_file(), "Root Dockerfile must exist"
    assert DOCKERIGNORE_PATH.is_file(), "Root .dockerignore must exist"


def test_dockerfile_multi_stage_structure(dockerfile_content: str):
    """Verify Dockerfile uses a multi-stage build with supported slim Python base."""
    from_matches = re.findall(
        r"^FROM\s+([^\s]+)(?:\s+AS\s+([^\s]+))?",
        dockerfile_content,
        re.MULTILINE | re.IGNORECASE,
    )
    assert len(from_matches) >= 2, "Dockerfile must define at least 2 build stages"

    builder_stage = from_matches[0]
    runtime_stage = from_matches[-1]

    # Validate builder stage
    assert re.search(r"python:3\.(1[0-9])-slim", builder_stage[0], re.IGNORECASE), (
        f"Builder base image must be a supported Python slim image, got: {builder_stage[0]}"
    )
    assert builder_stage[1].lower() == "builder", (
        f"First stage must be named 'builder', got: {builder_stage[1]}"
    )

    # Validate runtime stage
    assert re.search(r"python:3\.(1[0-9])-slim", runtime_stage[0], re.IGNORECASE), (
        f"Runtime base image must be a supported Python slim image, got: {runtime_stage[0]}"
    )
    assert runtime_stage[1].lower() == "runtime", (
        f"Final stage must be named 'runtime', got: {runtime_stage[1]}"
    )


def test_dockerfile_builder_stage_directives(dockerfile_content: str):
    """Verify builder stage installs build dependencies and compiles wheel."""
    # Split content by FROM to inspect builder stage specifically
    stages = re.split(r"^FROM\s+", dockerfile_content, flags=re.MULTILINE | re.IGNORECASE)
    assert len(stages) >= 3, "Failed to parse stages"
    builder_body = stages[1]

    assert re.search(r"hatchling", builder_body, re.IGNORECASE), (
        "Builder stage must install hatchling build backend"
    )
    assert re.search(r"python\s+-m\s+build\s+--wheel", builder_body), (
        "Builder stage must build a wheel using 'python -m build --wheel'"
    )
    assert re.search(r"COPY\s+.*?pyproject\.toml", builder_body), (
        "Builder stage must copy pyproject.toml"
    )
    assert re.search(r"COPY\s+src/", builder_body), "Builder stage must copy source directory src/"


def test_dockerfile_runtime_stage_security(dockerfile_content: str):
    """Verify runtime stage runs as non-root user devopsai with UID 10001."""
    stages = re.split(r"^FROM\s+", dockerfile_content, flags=re.MULTILINE | re.IGNORECASE)
    runtime_body = stages[-1]

    # Check non-root user creation with UID 10001
    assert re.search(r"10001", runtime_body), (
        "Runtime stage must explicitly configure UID/GID 10001"
    )
    assert re.search(r"devopsai", runtime_body), "Runtime stage must create user named 'devopsai'"
    assert re.search(r"USER\s+(?:10001:10001|devopsai:devopsai|10001|devopsai)", runtime_body), (
        "Runtime stage must switch to non-root user before execution"
    )

    # Verify wheel copied from builder and installed
    assert re.search(r"COPY\s+--from=builder\s+/build/dist/\*\.whl", runtime_body), (
        "Runtime stage must copy wheels from builder stage"
    )
    assert re.search(r"pip\s+install\s+.*?\.whl", runtime_body), (
        "Runtime stage must install the wheel via pip"
    )


def test_dockerfile_entrypoint_and_cmd(dockerfile_content: str):
    """Verify container default entrypoint is devops-ai."""
    entrypoint_match = re.search(r'ENTRYPOINT\s+\[\s*"devops-ai"\s*\]', dockerfile_content)
    assert entrypoint_match is not None, 'Dockerfile must specify ENTRYPOINT ["devops-ai"]'

    cmd_match = re.search(r'CMD\s+\[\s*"--help"\s*\]', dockerfile_content)
    assert cmd_match is not None, 'Dockerfile must specify CMD ["--help"]'


def test_dockerfile_healthcheck_directive(dockerfile_content: str):
    """Verify HEALTHCHECK runs the existing CLI health check command."""
    # Collapse line continuations for robust directive inspection
    normalized = re.sub(r"\\\s*\r?\n\s*", " ", dockerfile_content)

    healthcheck_match = re.search(
        r'HEALTHCHECK\s+.*?CMD\s+(?:\[\s*"devops-ai",\s*"health",\s*"check"\s*\]|devops-ai\s+health\s+check)',
        normalized,
    )
    assert healthcheck_match is not None, (
        "Dockerfile must define HEALTHCHECK invoking 'devops-ai health check'"
    )
    assert "--interval=" in normalized, "HEALTHCHECK must declare --interval"
    assert "--timeout=" in normalized, "HEALTHCHECK must declare --timeout"
    assert "--retries=" in normalized, "HEALTHCHECK must declare --retries"


def test_dockerfile_no_secrets_embedded(dockerfile_content: str):
    """Verify no API keys, credentials, or .env copies are embedded in Dockerfile."""
    assert not re.search(r"AIza[0-9A-Za-z-_]{20,}", dockerfile_content), (
        "No Google API keys should be embedded in Dockerfile"
    )
    assert not re.search(r"COPY\s+\.env\b", dockerfile_content), (
        "Dockerfile must not copy local .env file"
    )
    assert not re.search(r"ADD\s+\.env\b", dockerfile_content), (
        "Dockerfile must not add local .env file"
    )


def test_dockerfile_container_environment_defaults(dockerfile_content: str):
    """Verify container environment variables set appropriate production defaults."""
    assert "APP_ENV=production" in dockerfile_content
    assert "METRICS_HOST=" in dockerfile_content
    assert "0.0.0.0" in dockerfile_content
    assert "METRICS_PORT=" in dockerfile_content
    assert "9090" in dockerfile_content
    assert "EXPOSE 9090" in dockerfile_content


def test_dockerignore_critical_exclusions(dockerignore_lines: list[str]):
    """Verify .dockerignore excludes sensitive and temporary files."""
    patterns = set(dockerignore_lines)

    # Check git exclusion
    assert any(p.startswith(".git") for p in patterns), ".git must be ignored"

    # Check python caches
    assert any("__pycache__" in p for p in patterns), "__pycache__ must be ignored"
    assert any("*.py" in p for p in patterns), "Compiled python bytecode must be ignored"

    # Check test caches
    assert any(".pytest_cache" in p for p in patterns), ".pytest_cache must be ignored"
    assert any(".ruff_cache" in p for p in patterns), ".ruff_cache must be ignored"
    assert any(".coverage" in p for p in patterns), ".coverage must be ignored"

    # Check virtual environments
    assert any(".venv" in p for p in patterns), ".venv must be ignored"
    assert any("venv" in p for p in patterns), "venv must be ignored"

    # Check secrets
    assert any(p == ".env" or p.startswith(".env") for p in patterns), ".env files must be ignored"

    # Check build artifacts
    assert any("dist" in p for p in patterns), "dist/ build artifacts must be ignored"
    assert any("build" in p for p in patterns), "build/ artifacts must be ignored"


def test_dockerignore_preserves_required_packaging_assets(dockerignore_lines: list[str]):
    """Verify .dockerignore does not exclude files needed to build the distribution wheel."""
    required_assets = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "NOTICE",
        "src/devops_ai/__init__.py",
        "src/devops_ai/cli.py",
        "src/devops_ai/finops/agent.py",
    ]

    for asset in required_assets:
        for rule in dockerignore_lines:
            # Skip negation patterns or comments
            if rule.startswith("!"):
                continue
            clean_rule = rule.rstrip("/")
            # Check direct match or directory match
            matched = fnmatch.fnmatch(asset, clean_rule) or fnmatch.fnmatch(
                asset, f"{clean_rule}/*"
            )
            assert not matched, (
                f"Required packaging asset '{asset}' is accidentally excluded "
                f"by .dockerignore rule '{rule}'"
            )


def test_cli_health_check_command_is_registered():
    """Verify that 'devops-ai health check' is an actual registered CLI command."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["health", "check", "--help"])
    assert result.exit_code == 0, (
        f"Expected 'devops-ai health check --help' to succeed, got: {result.output}"
    )
    assert "Evaluate process liveness and readiness probes" in result.output
