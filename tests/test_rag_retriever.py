# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying semantic retrieval, cosine similarity ranking, thresholding, and provenance."""

from unittest.mock import patch
import pytest

from devops_ai.core.exceptions import RetrievalError
from devops_ai.rag.embeddings import MockEmbeddingProvider
from devops_ai.rag.models import DocumentChunk, RetrievalResult
from devops_ai.rag.retriever import RAGRetriever
from devops_ai.rag.vector_store import ChromaVectorStore


def _build_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    filename: str,
    content_hash: str = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        chunk_index=0,
        total_chunks=1,
        content=text,
        character_count=len(text),
        content_hash=content_hash or f"hash-sha256-retriever-{chunk_id}",
        metadata={"source_filename": filename, "source_path": f"/docs/{filename}"},
    )


def test_retriever_query_and_cosine_ranking():
    """Verify semantic retrieval ranks most similar chunks first with accurate provenance."""
    provider = MockEmbeddingProvider(dimension=32)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_retriever_rank", dimension=32
    )

    c1 = _build_chunk(
        "CHK-01", "DOC-K8S", "Kubernetes cluster pod crashloop investigation", "k8s_guide.md"
    )
    c2 = _build_chunk(
        "CHK-02", "DOC-NET", "VPC peering latency and DNS resolution timeout", "network_guide.md"
    )
    c3 = _build_chunk(
        "CHK-03", "DOC-AUTH", "OAuth2 authorization bearer token refresh", "auth_guide.md"
    )

    # Ingest into store with mock embeddings
    embeddings = provider.embed_batch([c1.content, c2.content, c3.content])
    store.add_chunks([c1, c2, c3], embeddings)

    retriever = RAGRetriever(
        embedding_provider=provider,
        vector_store=store,
    )

    # Search for Kubernetes topic with score_threshold=0.0 to verify ranking
    results = retriever.retrieve(
        "Kubernetes cluster pod crashloop investigation",
        top_k=2,
        score_threshold=0.0,
    )
    assert len(results) == 2

    # Top result should match CHK-01 with exact similarity = 1.0
    top = results[0]
    assert isinstance(top, RetrievalResult)
    assert top.chunk_id == "CHK-01"
    assert top.document_id == "DOC-K8S"
    assert top.source_filename == "k8s_guide.md"
    assert top.source_path == "/docs/k8s_guide.md"
    assert "Kubernetes cluster" in top.content
    assert pytest.approx(top.similarity_score, 1e-4) == 1.0


def test_retriever_score_threshold_and_top_k():
    """Verify score thresholding filters out irrelevant chunks and respects top_k limit."""
    provider = MockEmbeddingProvider(dimension=16)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_retriever_thresh", dimension=16
    )

    c1 = _build_chunk("C1", "D1", "Exact query match text", "doc1.md")
    c2 = _build_chunk("C2", "D2", "Completely unrelated topic text", "doc2.md")

    embeddings = provider.embed_batch([c1.content, c2.content])
    store.add_chunks([c1, c2], embeddings)

    retriever = RAGRetriever(
        embedding_provider=provider,
        vector_store=store,
    )

    # With high threshold (0.95), only exact match returns
    high_threshold_results = retriever.retrieve(
        "Exact query match text",
        top_k=5,
        score_threshold=0.95,
    )
    assert len(high_threshold_results) == 1
    assert high_threshold_results[0].chunk_id == "C1"

    # Top-K constraint
    all_results = retriever.retrieve("Exact query match text", top_k=1, score_threshold=0.0)
    assert len(all_results) == 1


def test_retriever_deterministic_ordering_and_tie_breaking():
    """Verify retriever orders results deterministically using chunk_id on identical scores."""
    provider = MockEmbeddingProvider(dimension=8)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_retriever_ordering", dimension=8
    )

    # Identical vectors yield identical similarity scores
    c_b = _build_chunk("CHK-B", "DOC-1", "Duplicate text content", "doc_b.md")
    c_a = _build_chunk("CHK-A", "DOC-1", "Duplicate text content", "doc_a.md")

    vec = provider.embed_text("Duplicate text content")
    store.add_chunks([c_b, c_a], [vec, vec])

    retriever = RAGRetriever(
        embedding_provider=provider,
        vector_store=store,
    )

    # Deduplicate disabled to inspect tie-breaking ordering
    results = retriever.retrieve(
        "Duplicate text content",
        top_k=2,
        deduplicate_content=False,
    )
    assert len(results) == 2
    # Both scores are identical (~1.0), so chunk_id ascending breaks tie: CHK-A before CHK-B
    assert results[0].chunk_id == "CHK-A"
    assert results[1].chunk_id == "CHK-B"


def test_retriever_content_deduplication():
    """Verify deduplicate_content collapses chunks sharing identical content hashes."""
    provider = MockEmbeddingProvider(dimension=8)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_retriever_dedup", dimension=8
    )

    c1 = _build_chunk(
        "CHK-1",
        "DOC-ORIG",
        "Redundant identical runbook text",
        "runbook1.md",
        content_hash="hash-identical-12345",
    )
    c2 = _build_chunk(
        "CHK-2",
        "DOC-COPY",
        "Redundant identical runbook text",
        "runbook2.md",
        content_hash="hash-identical-12345",
    )

    vec = provider.embed_text("Redundant identical runbook text")
    store.add_chunks([c1, c2], [vec, vec])

    retriever = RAGRetriever(
        embedding_provider=provider,
        vector_store=store,
    )

    # Default deduplicate_content=True collapses to 1 result
    results = retriever.retrieve("Redundant identical runbook text", top_k=5)
    assert len(results) == 1


def test_retriever_empty_query_and_error_handling():
    """Verify empty queries return empty results and unexpected exceptions raise RetrievalError."""
    provider = MockEmbeddingProvider(dimension=8)
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_retriever_err", dimension=8
    )

    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   \n\t  ") == []
    assert retriever.retrieve("valid query", top_k=0) == []

    # Unexpected query failure wraps in RetrievalError
    with patch.object(store, "similarity_search", side_effect=RuntimeError("Index corrupt")):
        with pytest.raises(RetrievalError) as exc_info:
            retriever.retrieve("Diagnostic query")
        assert "Index corrupt" in str(exc_info.value)
