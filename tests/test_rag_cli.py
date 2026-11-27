# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""CLI integration tests for `devops-ai rag evaluate` command."""

import json
from pathlib import Path
import re
import pytest
from typer.testing import CliRunner

from devops_ai.cli import app
from devops_ai.rag.embeddings.mock import MockEmbeddingProvider
from devops_ai.rag.models import DocumentChunk
from devops_ai.rag.vector_store.chroma import ChromaVectorStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_rag_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up mock embedding and chroma persistent store for CLI tests."""
    db_path = str(tmp_path / "cli_chroma_db")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RAG_VECTOR_STORE", "chroma")
    monkeypatch.setenv("RAG_VECTOR_STORE_PATH", db_path)
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.0")

    # Pre-seed vector store with a chunk so retriever has data
    store = ChromaVectorStore(persist_directory=db_path, dimension=768)
    provider = MockEmbeddingProvider(dimension=768)
    chunk = DocumentChunk(
        chunk_id="DOC-K8S-RUNBOOK-01-C0",
        document_id="DOC-K8S-RUNBOOK-01",
        chunk_index=0,
        total_chunks=1,
        content="Kubernetes pod CrashLoopBackOff OOMKilled exit code 137 container memory limit remediation runbook.",
        character_count=100,
        content_hash="hash_k8s_01_c0_0123456789",
        metadata={"source_filename": "k8s_runbook.md", "source_path": "/docs/k8s_runbook.md"},
    )
    store.add_chunks([chunk], provider.embed_batch([chunk.content]))


def _clean_output(text: str) -> str:
    """Strip ANSI escape sequences from CLI output for robust substring assertions."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def test_cli_rag_help():
    result = runner.invoke(app, ["rag", "--help"])
    assert result.exit_code == 0
    assert "evaluate" in _clean_output(result.output)


def test_cli_rag_evaluate_help():
    result = runner.invoke(app, ["rag", "evaluate", "--help"])
    assert result.exit_code == 0
    clean = _clean_output(result.output)
    assert "--dataset" in clean
    assert "--min-precision" in clean
    assert "--json" in clean


def test_cli_rag_evaluate_builtin_human_readable():
    result = runner.invoke(app, ["rag", "evaluate", "--k", "1,3"])
    assert result.exit_code == 0
    assert "Retrieval Quality Metrics" in result.output
    assert "Retrieval Latency Observability" in result.output
    assert "Quality Gates" in result.output


def test_cli_rag_evaluate_json():
    result = runner.invoke(app, ["rag", "evaluate", "--k", "1,3", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["dataset_id"] == "builtin-devops-v1"
    assert "mean_retrieval_metrics" in parsed
    assert "latency_summary" in parsed
    assert parsed["total_queries"] == 10


def test_cli_rag_evaluate_output_file(tmp_path: Path):
    out_file = tmp_path / "report.json"
    result = runner.invoke(app, ["rag", "evaluate", "--k", "1,3", "--output", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert parsed["dataset_id"] == "builtin-devops-v1"


def test_cli_rag_evaluate_threshold_failure():
    # Setting an impossible precision threshold of 1.0 against built-in queries when only 1 chunk is seeded
    result = runner.invoke(app, ["rag", "evaluate", "--k", "1,3", "--min-precision", "1.0"])
    assert result.exit_code == 1
    assert (
        "Quality Gates Failed" in result.output
        or "Quality gate threshold violation" in result.output
    )


def test_cli_rag_evaluate_missing_dataset(tmp_path: Path):
    result = runner.invoke(app, ["rag", "evaluate", "--dataset", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "Dataset Validation Error" in result.output
