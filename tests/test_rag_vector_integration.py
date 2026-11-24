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


def test_pipeline_reingest_changed_file_removes_obsolete_chunks(tmp_path: Path):
    """Verify modifying a file and re-ingesting it purges obsolete chunks and keeps store clean."""
    provider = MockEmbeddingProvider(dimension=16)
    store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test_reingest_obsolete",
        dimension=16,
    )
    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        vector_store=store,
        chunk_size=100,
        chunk_overlap=10,
    )

    doc_file = tmp_path / "scaling_runbook.md"
    # Version 1: 3 distinct sections producing 3 chunks
    v1_content = (
        "# Section Alpha\nDetailed instructions for alpha service scaling.\n\n"
        "# Section Beta\nDetailed instructions for beta service scaling.\n\n"
        "# Section Gamma\nDetailed instructions for gamma service scaling.\n"
    )
    doc_file.write_text(v1_content, encoding="utf-8")

    res1 = pipeline.ingest_file(doc_file)
    assert res1.status == IngestionStatus.SUCCESS
    assert res1.total_chunks == 3
    assert store.count() == 3

    # Version 2: Modified and condensed to a single section producing 1 chunk
    v2_content = "# Section Alpha Only\nCondensed single step scaling guide.\n"
    doc_file.write_text(v2_content, encoding="utf-8")

    res2 = pipeline.ingest_file(doc_file)
    assert res2.status == IngestionStatus.SUCCESS
    assert res2.total_chunks == 1
    # Obsolete chunks from version 1 must be deleted; only version 2 (1 chunk) remains
    assert store.count() == 1

    # Only current version is retrievable
    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)
    results = retriever.retrieve(query=res2.chunks[0].content, top_k=5, score_threshold=0.0)
    assert len(results) == 1
    assert results[0].document_id == res2.document_id
    assert "Condensed single step" in results[0].content


def test_pipeline_replacement_failure_preserves_previous_version(tmp_path: Path):
    """Verify that if re-ingesting an updated file fails, the previous valid version is preserved."""
    provider = MockEmbeddingProvider(dimension=16)
    store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test_preserve_prev",
        dimension=16,
    )
    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        vector_store=store,
        chunk_size=200,
    )

    doc_file = tmp_path / "critical_guide.md"
    doc_file.write_text("# Critical Runbook\nValid production procedures.", encoding="utf-8")

    res1 = pipeline.ingest_file(doc_file)
    assert res1.status == IngestionStatus.SUCCESS
    initial_count = store.count()
    assert initial_count >= 1

    # Update file content
    doc_file.write_text("# Updated Runbook\nNew replacement text.", encoding="utf-8")

    # Simulate failure during batch embedding of the updated file
    with patch.object(
        provider,
        "embed_batch",
        side_effect=EmbeddingProviderError("Ollama failed", provider="mock"),
    ):
        res2 = pipeline.ingest_file(doc_file, raise_on_error=False)
        assert res2.status == IngestionStatus.FAILED

    # Previous valid chunks must still be intact in store
    assert store.count() == initial_count
    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)
    results = retriever.retrieve(query=res1.chunks[0].content, top_k=2, score_threshold=0.0)
    assert len(results) >= 1
    assert results[0].document_id == res1.document_id


def test_pipeline_same_filename_different_paths(tmp_path: Path):
    """Verify files with identical names in different directories do not overwrite or delete each other."""
    provider = MockEmbeddingProvider(dimension=16)
    store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test_diff_paths",
        dimension=16,
    )
    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        vector_store=store,
        chunk_size=300,
    )

    dir_a = tmp_path / "service_a"
    dir_b = tmp_path / "service_b"
    dir_a.mkdir()
    dir_b.mkdir()

    file_a = dir_a / "deploy.md"
    file_b = dir_b / "deploy.md"
    file_a.write_text("# Service A Deploy\nDeploy steps for Service A.", encoding="utf-8")
    file_b.write_text("# Service B Deploy\nDeploy steps for Service B.", encoding="utf-8")

    res_a = pipeline.ingest_file(file_a)
    res_b = pipeline.ingest_file(file_b)
    assert res_a.status == IngestionStatus.SUCCESS
    assert res_b.status == IngestionStatus.SUCCESS
    assert store.count() == 2

    # Update file A only
    file_a.write_text(
        "# Service A Deploy v2\nUpdated deploy steps for Service A.", encoding="utf-8"
    )
    res_a2 = pipeline.ingest_file(file_a)
    assert res_a2.status == IngestionStatus.SUCCESS

    # Total count remains 2 (1 for A v2, 1 for B)
    assert store.count() == 2

    # Verify B was not deleted
    chunk_b = store.get_chunk(res_b.chunks[0].chunk_id)
    assert chunk_b is not None
    assert "Service B Deploy" in chunk_b.content


def test_pipeline_failures_at_all_stages_and_rollback(tmp_path: Path):
    """Verify rollback removes inserted vectors and restores hash registry across all failure modes."""
    from devops_ai.core.exceptions import DocumentFormatError, VectorStoreError

    provider = MockEmbeddingProvider(dimension=16)
    store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test_stages_fail",
        dimension=16,
    )
    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        vector_store=store,
    )

    doc_file = tmp_path / "test_stage.md"
    doc_file.write_text("# Staging Doc\nTest content for stage failures.", encoding="utf-8")

    # Stage 1: Failure before embedding (e.g. format extractor error)
    with patch.object(
        pipeline.chunker, "chunk_document", side_effect=DocumentFormatError("Bad formatting")
    ):
        res1 = pipeline.ingest_file(doc_file, raise_on_error=False)
        assert res1.status == IngestionStatus.FAILED
        assert store.count() == 0
        # Hash was restored: re-ingest without patch should not be skipped as duplicate
        assert not pipeline.preprocessor.is_duplicate(
            pipeline.preprocessor.compute_hash(doc_file.read_text())
        )

    # Stage 2: Failure during vector insertion (add_chunks raises VectorStoreError)
    with patch.object(
        store, "add_chunks", side_effect=VectorStoreError("Chroma locked", store_name="chroma")
    ):
        res2 = pipeline.ingest_file(doc_file, raise_on_error=False)
        assert res2.status == IngestionStatus.FAILED
        assert store.count() == 0

    # Stage 3: Failure after vector insertion (e.g. post-indexing exception)
    original_index_chunks = pipeline._index_chunks

    def broken_index_and_raise(chunks):
        _ = original_index_chunks(chunks)
        raise RuntimeError("Disk full during post-processing")

    with patch.object(pipeline, "_index_chunks", side_effect=broken_index_and_raise):
        res3 = pipeline.ingest_file(doc_file, raise_on_error=False)
        assert res3.status == IngestionStatus.FAILED
        # Newly inserted vectors must be rolled back by delete_document
        assert store.count() == 0

    # Stage 4: In-memory ingestion failure (ingest_text)
    with patch.object(
        provider, "embed_batch", side_effect=EmbeddingProviderError("Batch error", provider="mock")
    ):
        res_text = pipeline.ingest_text(
            content="Some in memory text runbook",
            source_name="mem_runbook",
            raise_on_error=False,
        )
        assert res_text.status == IngestionStatus.FAILED
        assert store.count() == 0

    # Hash was unregistered: repeating without failure succeeds
    res_text_ok = pipeline.ingest_text(
        content="Some in memory text runbook",
        source_name="mem_runbook",
        raise_on_error=True,
    )
    assert res_text_ok.status == IngestionStatus.SUCCESS
    assert store.count() >= 1
