# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Tests for GitHub Container Registry (GHCR) publishing workflow and release configuration."""

from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH_WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "publish.yml"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def publish_workflow() -> dict:
    """Parse and return .github/workflows/publish.yml data."""
    assert PUBLISH_WORKFLOW_FILE.is_file(), (
        f"Publish workflow file not found at {PUBLISH_WORKFLOW_FILE}"
    )
    raw = PUBLISH_WORKFLOW_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "publish.yml must parse to a dictionary"
    return data


@pytest.fixture(scope="module")
def compose_data() -> dict:
    """Parse and return docker-compose.yml data."""
    assert COMPOSE_FILE.is_file(), f"docker-compose.yml not found at {COMPOSE_FILE}"
    raw = COMPOSE_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "docker-compose.yml must parse to a dictionary"
    return data


def test_publish_workflow_triggers(publish_workflow: dict):
    """Verify publish workflow triggers on main, v* tags, dispatch, and NEVER on pull_request."""
    triggers = publish_workflow.get("on") or publish_workflow.get(True)
    assert triggers is not None, "Workflow must define 'on' triggers"

    if isinstance(triggers, dict):
        # Push triggers
        assert "push" in triggers, "Workflow must trigger on 'push'"
        push_cfg = triggers["push"]
        branches = push_cfg.get("branches", [])
        assert "main" in branches, "Push trigger must include 'main' branch"

        tags = push_cfg.get("tags", [])
        assert any("v*" in tag for tag in tags), "Push trigger must include 'v*' version tags"

        # Manual trigger
        assert "workflow_dispatch" in triggers, "Workflow must support manual 'workflow_dispatch'"

        # Strictly forbid pull_request publishing
        assert "pull_request" not in triggers, (
            "Publish workflow must NEVER trigger on pull_request events"
        )


def test_publish_workflow_permissions_and_concurrency(publish_workflow: dict):
    """Verify minimal permissions (contents: read, packages: write) and concurrency."""
    perms = publish_workflow.get("permissions", {})
    assert perms.get("contents") == "read", "Workflow must restrict contents permission to 'read'"
    assert perms.get("packages") == "write", (
        "Workflow requires 'packages: write' to publish to GHCR"
    )

    # Ensure no overly broad permissions
    assert len(perms) == 2, (
        f"Workflow should only define contents and packages permissions, got: {perms}"
    )

    assert "concurrency" in publish_workflow, "Workflow must define concurrency management"


def test_publish_job_steps_and_actions(publish_workflow: dict):
    """Verify publish job configures Buildx, GHCR login, metadata extraction, and build-push."""
    assert "jobs" in publish_workflow, "Workflow must define 'jobs'"
    jobs = publish_workflow["jobs"]
    assert "publish" in jobs, "Workflow must define 'publish' job"

    publish_job = jobs["publish"]
    steps = publish_job.get("steps", [])
    step_uses = [s.get("uses", "") for s in steps if "uses" in s]

    # Checkout
    assert any("actions/checkout" in use for use in step_uses), "Must use actions/checkout"

    # Buildx
    assert any("docker/setup-buildx-action" in use for use in step_uses), (
        "Must configure docker/setup-buildx-action"
    )

    # Login to GHCR
    login_step = next((s for s in steps if "docker/login-action" in s.get("uses", "")), None)
    assert login_step is not None, "Must include docker/login-action step"
    login_with = login_step.get("with", {})
    assert login_with.get("registry") == "ghcr.io", "Registry must be 'ghcr.io'"
    assert "${{ secrets.GITHUB_TOKEN }}" in str(login_with.get("password")), (
        "Must authenticate using GITHUB_TOKEN secret"
    )

    # Metadata extraction
    meta_step = next((s for s in steps if "docker/metadata-action" in s.get("uses", "")), None)
    assert meta_step is not None, "Must include docker/metadata-action step"
    meta_with = meta_step.get("with", {})
    images = str(meta_with.get("images", ""))
    assert "ghcr.io" in images and "github.repository" in images, (
        f"Metadata images must target ghcr.io/${{{{ github.repository }}}}, got: {images}"
    )

    # Build and Push
    build_step = next((s for s in steps if "docker/build-push-action" in s.get("uses", "")), None)
    assert build_step is not None, "Must include docker/build-push-action step"
    build_with = build_step.get("with", {})
    assert build_with.get("push") is True, "Build-push action must set push: true"
    assert build_with.get("context") == ".", "Build context must be root '.'"
    assert build_with.get("file") == "./Dockerfile", (
        "Build action must explicitly use file: './Dockerfile'"
    )


def test_docker_build_context_and_files():
    """Verify Dockerfile is at repository root and required packaging assets exist in build context."""
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.is_file(), "Dockerfile must exist at repository root"

    # Verify root packaging assets required by the Dockerfile builder stage
    required_assets = ["pyproject.toml", "README.md", "LICENSE", "NOTICE"]
    for asset_name in required_assets:
        asset_path = REPO_ROOT / asset_name
        assert asset_path.is_file(), f"Required build context file '{asset_name}' must exist"

    # Verify source directory
    src_dir = REPO_ROOT / "src" / "devops_ai"
    assert src_dir.is_dir(), "Required build context directory 'src/devops_ai' must exist"

    # Verify .dockerignore does not exclude required packaging assets
    dockerignore_file = REPO_ROOT / ".dockerignore"
    assert dockerignore_file.is_file(), ".dockerignore must exist at repository root"
    dockerignore_content = dockerignore_file.read_text(encoding="utf-8")
    for asset_name in required_assets:
        assert (
            f"\n{asset_name}\n" not in dockerignore_content
            and f"\n{asset_name}/" not in dockerignore_content
        ), f".dockerignore must not exclude '{asset_name}'"


def test_publish_metadata_tagging_strategy(publish_workflow: dict):
    """Verify Docker metadata tags include branch, semver, sha, and latest restricted to default branch.

    Policy:
    - Default-branch (main) builds publish latest (enable={{is_default_branch}}).
    - Version-tag (v*) builds publish semantic version tags and SHA tags.
    - Version-tag builds do NOT publish latest unless explicitly configured.
    """
    publish_job = publish_workflow["jobs"]["publish"]
    steps = publish_job.get("steps", [])
    meta_step = next((s for s in steps if "docker/metadata-action" in s.get("uses", "")), None)
    assert meta_step is not None

    tags_raw = meta_step.get("with", {}).get("tags", "")
    assert "type=ref,event=branch" in tags_raw, "Must generate branch ref tag"
    assert "type=semver,pattern={{version}}" in tags_raw, "Must generate full semantic version tag"
    assert "type=semver,pattern={{major}}.{{minor}}" in tags_raw, (
        "Must generate major.minor semantic version tag"
    )
    assert "type=semver,pattern={{major}}" in tags_raw, "Must generate major semantic version tag"
    assert "type=sha,prefix=sha-,format=short" in tags_raw, "Must generate short git SHA tag"
    assert "type=raw,value=latest,enable={{is_default_branch}}" in tags_raw, (
        "Must configure latest tag strictly with 'enable={{is_default_branch}}' so version-tag builds do not publish latest"
    )


def test_no_hardcoded_secrets_in_publish_workflow():
    """Verify publish workflow does not contain hardcoded credentials or personal tokens."""
    raw = PUBLISH_WORKFLOW_FILE.read_text(encoding="utf-8")
    assert "ghp_" not in raw, "Personal access token detected in publish workflow"
    assert "github_pat_" not in raw, "Fine-grained personal access token detected"
    assert "password:" in raw
    assert "${{ secrets.GITHUB_TOKEN }}" in raw, "Must use standard GITHUB_TOKEN"


def test_compose_published_image_compatibility(compose_data: dict):
    """Verify docker-compose.yml supports DEVOPS_AI_IMAGE override while defaulting to local build."""
    svc = compose_data["services"]["devops-ai"]
    raw_compose = COMPOSE_FILE.read_text(encoding="utf-8")

    # Assert environment variable override pattern in raw YAML
    assert "DEVOPS_AI_IMAGE" in raw_compose, (
        "docker-compose.yml must support DEVOPS_AI_IMAGE environment variable"
    )
    assert "${DEVOPS_AI_IMAGE:-devops-ai:local}" in raw_compose, (
        "docker-compose.yml must default to 'devops-ai:local'"
    )

    # Verify build configuration is preserved
    assert "build" in svc, "Build configuration must remain in docker-compose.yml"
    assert svc["build"]["context"] == ".", "Build context must remain '.'"
