# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for DockerfileGenerator and static heuristics."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.generators.dockerfile import DockerfileGenerator
from devops_ai.llm.base import BaseLLMProvider


@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    provider.provider_name = "ollama"
    provider.model_name = "llama3.2:1b"
    return provider


def test_dockerfile_non_root_user_success(mock_provider):
    content = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
USER app
CMD ["python", "main.py"]
"""
    mock_provider.generate.return_value = content
    generator = DockerfileGenerator(provider=mock_provider)
    result = generator.generate(language="python", port=8000)

    assert result.is_valid is True
    assert result.content == content.strip()
    assert result.provider == "ollama"
    assert result.model == "llama3.2:1b"
    root_warnings = [w for w in result.warnings if "root" in w.lower()]
    assert len(root_warnings) == 0


def test_dockerfile_root_user_warnings(mock_provider):
    root_variants = ["root", "0", "00", "0:0", "00:00", "root:root", "0:root", "root:0"]
    generator = DockerfileGenerator(provider=mock_provider)

    for variant in root_variants:
        content = f"""FROM python:3.12-slim
EXPOSE 8000
USER {variant}
CMD ["python", "main.py"]
"""
        mock_provider.generate.return_value = content
        result = generator.generate(language="python", port=8000)
        assert any("explicitly configured with root user" in w.lower() for w in result.warnings)


def test_dockerfile_missing_user_warning(mock_provider):
    content = """FROM python:3.12-slim
EXPOSE 8000
CMD ["python", "main.py"]
"""
    mock_provider.generate.return_value = content
    generator = DockerfileGenerator(provider=mock_provider)
    result = generator.generate(language="python", port=8000)
    assert any("no 'user' instruction found" in w.lower() for w in result.warnings)


def test_dockerfile_multistage_user_evaluation(mock_provider):
    # Stage 1 runs as root, final stage sets USER app -> NO root warnings
    content_safe = """FROM golang:1.22 AS builder
WORKDIR /src
COPY . .
RUN go build -o app

FROM alpine:3.19
EXPOSE 8080
COPY --from=builder /src/app /app
USER app
CMD ["/app"]
"""
    mock_provider.generate.return_value = content_safe
    generator = DockerfileGenerator(provider=mock_provider)
    result = generator.generate(language="golang", port=8080)
    root_warnings = [w for w in result.warnings if "root" in w.lower()]
    assert len(root_warnings) == 0

    # Stage 1 sets USER app, but final stage misses USER -> root warning emitted
    content_unsafe = """FROM golang:1.22 AS builder
WORKDIR /src
USER app
COPY . .
RUN go build -o app

FROM alpine:3.19
EXPOSE 8080
COPY --from=builder /src/app /app
CMD ["/app"]
"""
    mock_provider.generate.return_value = content_unsafe
    result = generator.generate(language="golang", port=8080)
    assert any("no 'user' instruction found in final build stage" in w.lower() for w in result.warnings)


def test_dockerfile_comment_stripping_ignores_fake_user(mock_provider):
    # Commented-out USER app must NOT be counted as active non-root user
    content = """FROM python:3.12-slim
EXPOSE 8000
# USER app
# USER 1000
CMD ["python", "main.py"]
"""
    mock_provider.generate.return_value = content
    generator = DockerfileGenerator(provider=mock_provider)
    result = generator.generate(language="python", port=8000)
    assert any("no 'user' instruction found" in w.lower() for w in result.warnings)


def test_dockerfile_rejects_boolean_port(mock_provider):
    generator = DockerfileGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(language="python", port=True)  # type: ignore[arg-type]
    assert exc_info.value.code == "INVALID_ARGUMENT"
    assert "booleans rejected" in str(exc_info.value)


def test_dockerfile_rejects_out_of_range_port(mock_provider):
    generator = DockerfileGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(language="python", port=70000)
    assert exc_info.value.code == "INVALID_ARGUMENT"

    with pytest.raises(DevOpsAIError) as exc_info2:
        generator.generate(language="python", port=0)
    assert exc_info2.value.code == "INVALID_ARGUMENT"


def test_dockerfile_fatal_syntax_missing_from(mock_provider):
    mock_provider.generate.return_value = "RUN pip install -r requirements.txt\nCMD python app.py"
    generator = DockerfileGenerator(provider=mock_provider)
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(language="python", port=8000)
    assert exc_info.value.code == "GENERATOR_VALIDATION_ERROR"
    assert "missing required 'FROM'" in str(exc_info.value)


def test_dockerfile_fatal_destructive_commands(mock_provider):
    generator = DockerfileGenerator(provider=mock_provider)

    mock_provider.generate.return_value = "FROM python:3.12-slim\nRUN rm -rf /\nUSER app"
    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(language="python", port=8000)
    assert exc_info.value.code == "GENERATOR_SECURITY_VIOLATION"

    mock_provider.generate.return_value = "FROM python:3.12-slim\nRUN chmod 777 /app\nUSER app"
    # Notice: chmod 777 /app is permissive on /app, but chmod 777 / is root violation
    mock_provider.generate.return_value = "FROM python:3.12-slim\nRUN chmod 777 /\nUSER app"
    with pytest.raises(DevOpsAIError) as exc_info2:
        generator.generate(language="python", port=8000)
    assert exc_info2.value.code == "GENERATOR_SECURITY_VIOLATION"


def test_dockerfile_refuse_overwrite_without_force(mock_provider, tmp_path: Path):
    target = tmp_path / "Dockerfile"
    target.write_text("EXISTING CONTENT", encoding="utf-8")

    mock_provider.generate.return_value = "FROM python:3.12-slim\nEXPOSE 8000\nUSER app"
    generator = DockerfileGenerator(provider=mock_provider)

    with pytest.raises(DevOpsAIError) as exc_info:
        generator.generate(language="python", port=8000, output_path=target, force=False)
    assert exc_info.value.code == "FILE_ALREADY_EXISTS"
    assert target.read_text(encoding="utf-8") == "EXISTING CONTENT"


def test_dockerfile_atomic_overwrite_with_force(mock_provider, tmp_path: Path):
    target = tmp_path / "Dockerfile"
    target.write_text("EXISTING CONTENT", encoding="utf-8")

    mock_provider.generate.return_value = "FROM python:3.12-slim\nEXPOSE 8000\nUSER app"
    generator = DockerfileGenerator(provider=mock_provider)

    result = generator.generate(language="python", port=8000, output_path=target, force=True)
    assert result.is_valid is True
    assert "FROM python:3.12-slim" in target.read_text(encoding="utf-8")


def test_dockerfile_markdown_fence_cleaning(mock_provider):
    fenced_output = """```dockerfile
FROM python:3.12-slim
EXPOSE 8000
USER app
```"""
    mock_provider.generate.return_value = fenced_output
    generator = DockerfileGenerator(provider=mock_provider)
    result = generator.generate(language="python", port=8000)
    assert not result.content.startswith("```")
    assert not result.content.endswith("```")
    assert result.content.startswith("FROM python:3.12-slim")


def test_dockerfile_port_expose_warning(mock_provider):
    content = "FROM python:3.12-slim\nEXPOSE 3000\nUSER app"
    mock_provider.generate.return_value = content
    generator = DockerfileGenerator(provider=mock_provider)
    result = generator.generate(language="python", port=8000)
    assert any("requested port 8000 does not appear in an expose instruction" in w.lower() for w in result.warnings)
