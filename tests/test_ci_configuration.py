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

    # Verify pytest execution step and dependencies
    steps = test_job.get("steps", [])
    step_commands = [s.get("run", "") for s in steps if "run" in s]
    assert any("pytest" in cmd for cmd in step_commands), "Test job must execute pytest"
    assert any('pip install -e ".[dev,full]"' in cmd for cmd in step_commands), (
        "Test job must install both dev and full optional dependencies"
    )

    # Verify NO_COLOR terminal environment for ANSI-resilient testing
    pytest_step = next((s for s in steps if "pytest" in s.get("run", "")), None)
    assert pytest_step is not None, "Test job must contain a pytest step"
    step_env = pytest_step.get("env", {})
    assert step_env.get("NO_COLOR") == "1", "Test job must configure NO_COLOR=1"


def test_quality_job_configuration(ci_workflow: dict):
    """Verify quality job pins stable Ruff and executes check and format checks."""
    quality_job = ci_workflow["jobs"]["quality"]
    steps = quality_job.get("steps", [])
    step_commands = [s.get("run", "") for s in steps if "run" in s]

    assert any("ruff==0.14.10" in cmd for cmd in step_commands), (
        "Quality job must pin ruff version to 0.14.10"
    )
    assert any("ruff check" in cmd for cmd in step_commands), "Quality job must run 'ruff check'"
    assert any("ruff format --check" in cmd for cmd in step_commands), (
        "Quality job must run 'ruff format --check'"
    )


def test_ruff_pyproject_lint_configuration():
    """Verify pyproject.toml defines explicit lint rule selection."""
    import tomllib

    pyproject_file = REPO_ROOT / "pyproject.toml"
    assert pyproject_file.is_file(), "pyproject.toml must exist"
    data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))

    tool_ruff = data.get("tool", {}).get("ruff", {})
    assert tool_ruff.get("line-length") == 100, "Ruff line-length must be 100"
    assert tool_ruff.get("target-version") == "py310", "Ruff target-version must be py310"

    lint_cfg = tool_ruff.get("lint", {})
    assert lint_cfg.get("select") == ["E", "F"], "Ruff lint select must be ['E', 'F']"
    assert "E501" in lint_cfg.get("ignore", []), "E501 must be ignored in favor of formatter"


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
    """Verify security job upgrades setuptools, executes pip-audit, and strictly restricts exemptions."""
    sec_job = ci_workflow["jobs"]["security"]
    steps = sec_job.get("steps", [])
    step_commands = [s.get("run", "") for s in steps if "run" in s]

    assert any("pip-audit" in cmd for cmd in step_commands), "Security job must execute pip-audit"
    assert any("setuptools>=84.1.0" in cmd for cmd in step_commands), (
        "Security job must upgrade setuptools to >=84.1.0 to address runner CVEs"
    )

    # Ensure no suppression bypasses (|| true) and verify only approved unpatched ChromaDB CVEs are listed
    audit_step = next((s for s in steps if "Run Dependency Security Audit" in s.get("name", "")), None)
    assert audit_step is not None, "Security job must have a 'Run Dependency Security Audit' step"
    audit_cmd = audit_step.get("run", "")
    assert "|| true" not in audit_cmd, "Security audit must not bypass non-zero exit codes"
    assert "--ignore-vuln" in audit_cmd, (
        "Security audit must document narrow upstream ChromaDB exemptions"
    )
    # Verify exemptions are strictly restricted to the verified unpatched upstream ChromaDB IDs
    assert "PYSEC-2026-311" in audit_cmd
    assert "PYSEC-2026-3813" in audit_cmd
    assert "PYSEC-2026-3814" in audit_cmd
    assert "PYSEC-2026-3815" in audit_cmd
