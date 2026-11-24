# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""End-to-end integration tests verifying document ingestion, embedding generation, vector indexing, and retrieval."""

from pathlib import Path
from unittest.mock import patch
import pytest

from devops_ai.core.exceptions import EmbeddingProviderError
from devops_ai.rag.embeddings import MockEmbeddingProvider
from devops_ai.rag.models import IngestionStatus
from devops_ai.rag.pipeline import DocumentIngestionPipeline
from devops_ai.rag.retriever import RAGRetriever
from devops_ai.rag.vector_store import ChromaVectorStore


def test_pipeline_end_to_end_indexing_and_retrieval(tmp_path: Path):
    """Verify document ingestion generates embeddings, stores them, and enables semantic retrieval."""
    provider = MockEmbeddingProvider(dimension=64)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="integ_end_to_end", dimension=64
    )

    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        vector_store=store,
        chunk_size=300,
        chunk_overlap=50,
    )

    doc_file = tmp_path / "k8s_recovery.md"
    content = (
        "# Kubernetes Cluster Recovery\n\n"
        "## Pod Eviction Procedures\n"
        "When nodes encounter DiskPressure, eviction manager drains non-critical pods.\n"
        "To inspect pod evictions, execute kubectl describe pod.\n"
    )
    doc_file.write_text(content, encoding="utf-8")

    # Ingest file
    result = pipeline.ingest_file(doc_file)
    assert result.status == IngestionStatus.SUCCESS
    assert result.total_chunks >= 1
    assert result.metadata["indexed_chunks"] == result.total_chunks
    assert result.metadata["vector_store"] == "chroma"
    assert result.metadata["embedding_provider"] == "mock"
    assert store.count() == result.total_chunks

    # Retrieve through RAGRetriever using the chunk's content for exact match test
    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)
    query = result.chunks[0].content
    retrieved = retriever.retrieve(query, top_k=2)

    assert len(retrieved) >= 1
    top_hit = retrieved[0]
    assert "DiskPressure" in top_hit.content
    assert top_hit.document_id == result.document_id
    assert top_hit.source_filename == "k8s_recovery.md"
    assert pytest.approx(top_hit.similarity_score, 1e-4) == 1.0


def test_pipeline_repeated_ingestion_and_updates(tmp_path: Path):
    """Verify duplicate documents are skipped without creating duplicate vector entries."""
    provider = MockEmbeddingProvider(dimension=32)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="integ_repeated", dimension=32
    )

    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        vector_store=store,
        skip_duplicates=True,
    )

    doc_file = tmp_path / "sop.md"
    doc_file.write_text(
        "# Standard SOP\nSteps to verify ingress gateway routing.", encoding="utf-8"
    )

    # First ingestion
    res1 = pipeline.ingest_file(doc_file)
    assert res1.status == IngestionStatus.SUCCESS
    initial_count = store.count()
    assert initial_count >= 1

    # Repeated identical ingestion is skipped; vector store count remains unchanged
    res2 = pipeline.ingest_file(doc_file)
    assert res2.status == IngestionStatus.SKIPPED
    assert store.count() == initial_count


def test_pipeline_embedding_failure_does_not_claim_success(tmp_path: Path):
    """Verify failures during embedding generation abort indexing and report failure."""
    provider = MockEmbeddingProvider(dimension=32)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="integ_fail", dimension=32
    )

    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        vector_store=store,
    )

    doc_file = tmp_path / "broken.md"
    doc_file.write_text("# Critical Runbook\nFailover details.", encoding="utf-8")

    # Simulate provider failure during batch embedding
    with patch.object(
        provider, "embed_batch", side_effect=EmbeddingProviderError("Ollama OOM", provider="mock")
    ):
        # When raise_on_error=False
        result = pipeline.ingest_file(doc_file, raise_on_error=False)
        assert result.status == IngestionStatus.FAILED
        assert "Ollama OOM" in result.error
        assert result.metadata["error_type"] == "EmbeddingProviderError"
        assert store.count() == 0  # No partial vector indexed

        # When raise_on_error=True
        with pytest.raises(EmbeddingProviderError):
            pipeline.ingest_file(doc_file, raise_on_error=True)
        assert store.count() == 0
