# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Tests for GitHub Actions CI pipeline configuration and quality gates."""

from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    """Parse and return .github/workflows/ci.yml data."""
    assert CI_WORKFLOW_FILE.is_file(), f"CI workflow file not found at {CI_WORKFLOW_FILE}"
    raw = CI_WORKFLOW_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "ci.yml must parse to a dictionary"
    return data


def test_ci_workflow_triggers_and_permissions(ci_workflow: dict):
    """Verify CI workflow triggers on push/PR to main, dispatch, and sets read permissions."""
    # Verify triggers - PyYAML parses 'on' as boolean True unless quoted or keyed as True
    triggers = ci_workflow.get("on") or ci_workflow.get(True)
    assert triggers is not None, "Workflow must define 'on' triggers"

    if isinstance(triggers, dict):
        assert "push" in triggers, "Workflow must trigger on 'push'"
        assert "main" in triggers["push"].get("branches", []), "Push must target 'main'"
        assert "pull_request" in triggers, "Workflow must trigger on 'pull_request'"
        assert "main" in triggers["pull_request"].get("branches", []), "PR must target 'main'"
        assert "workflow_dispatch" in triggers, "Workflow must allow manual 'workflow_dispatch'"

    # Verify least privilege permissions
    perms = ci_workflow.get("permissions", {})
    assert perms.get("contents") == "read", "Workflow must restrict permissions to 'contents: read'"

    # Verify concurrency control
    assert "concurrency" in ci_workflow, "Workflow must define concurrency controls"


def test_ci_jobs_declared(ci_workflow: dict):
    """Verify all four required CI jobs are declared."""
    assert "jobs" in ci_workflow, "Workflow must define 'jobs'"
    jobs = ci_workflow["jobs"]
    expected_jobs = {"quality", "test", "build", "security"}
    assert expected_jobs.issubset(set(jobs.keys())), (
        f"CI workflow missing expected jobs. Defined: {set(jobs.keys())}, expected: {expected_jobs}"
    )


def test_test_job_matrix_coverage(ci_workflow: dict):
    """Verify test job covers all supported Python versions: 3.10, 3.11, 3.12, 3.13."""
    test_job = ci_workflow["jobs"]["test"]
    strategy = test_job.get("strategy", {})
    matrix = strategy.get("matrix", {})
    versions = [str(v) for v in matrix.get("python-version", [])]

    assert "3.10" in versions, "Python 3.10 must be in test matrix"
    assert "3.11" in versions, "Python 3.11 must be in test matrix"
    assert "3.12" in versions, "Python 3.12 must be in test matrix"
    assert "3.13" in versions, "Python 3.13 must be in test matrix"

    # Verify pytest execution step
    steps = test_job.get("steps", [])
    step_commands = [s.get("run", "") for s in steps if "run" in s]
    assert any("pytest" in cmd for cmd in step_commands), "Test job must execute pytest"


def test_quality_job_configuration(ci_workflow: dict):
    """Verify quality job runs ruff check and ruff format --check."""
    quality_job = ci_workflow["jobs"]["quality"]
    steps = quality_job.get("steps", [])
    step_commands = [s.get("run", "") for s in steps if "run" in s]

    assert any("ruff check" in cmd for cmd in step_commands), "Quality job must run 'ruff check'"
    assert any("ruff format --check" in cmd for cmd in step_commands), (
        "Quality job must run 'ruff format --check'"
    )


def test_build_job_configuration(ci_workflow: dict):
    """Verify build job compiles wheel and uploads artifact."""
    build_job = ci_workflow["jobs"]["build"]
    steps = build_job.get("steps", [])
    step_commands = [s.get("run", "") for s in steps if "run" in s]

    assert any("build --wheel" in cmd for cmd in step_commands), (
        "Build job must execute 'python -m build --wheel'"
    )

    # Verify upload-artifact action
    action_uses = [s.get("uses", "") for s in steps if "uses" in s]
    assert any("actions/upload-artifact" in use for use in action_uses), (
        "Build job must use actions/upload-artifact to preserve wheels"
    )


def test_security_job_configuration(ci_workflow: dict):
    """Verify security job executes pip-audit on resolved dependencies."""
    sec_job = ci_workflow["jobs"]["security"]
    steps = sec_job.get("steps", [])
    step_commands = [s.get("run", "") for s in steps if "run" in s]

    assert any("pip-audit" in cmd for cmd in step_commands), "Security job must execute pip-audit"
    # Ensure no hardcoded ignore flags or bypasses
    for cmd in step_commands:
        if "pip-audit" in cmd:
            assert "--ignore-vuln" not in cmd, (
                "Security audit must not silently ignore vulnerabilities"
            )
            assert "|| true" not in cmd, "Security audit must not bypass non-zero exit codes"
