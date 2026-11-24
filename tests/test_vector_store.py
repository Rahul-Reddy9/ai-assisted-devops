# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying ChromaDB persistent vector storage, indexing, and recovery."""

from pathlib import Path
from unittest.mock import patch
import pytest

from devops_ai.core.exceptions import (
    EmbeddingDimensionMismatchError,
    VectorStoreError,
)
from devops_ai.rag.models import DocumentChunk
from devops_ai.rag.vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
    get_vector_store,
)


def _create_sample_chunk(chunk_id: str, doc_id: str, text: str, index: int = 0) -> DocumentChunk:
    """Helper to construct valid DocumentChunk instances for testing."""
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        chunk_index=index,
        total_chunks=1,
        content=text,
        character_count=len(text),
        content_hash=f"hash-sha256-content-{chunk_id}-{len(text)}",
        metadata={"source_filename": f"{doc_id.lower()}.md", "topic": "sre"},
    )


def test_chroma_vector_store_ephemeral_basic_operations():
    """Verify in-memory ephemeral ChromaVectorStore insertion, counting, and search."""
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_ephemeral_basic", dimension=4
    )
    assert isinstance(store, BaseVectorStore)
    assert store.count() == 0

    chunk1 = _create_sample_chunk("CHK-1", "DOC-A", "Kubernetes cluster outage runbook")
    chunk2 = _create_sample_chunk("CHK-2", "DOC-B", "Redis cache connection pool exhaustion")

    # Add chunks with orthogonal embeddings
    store.add_chunks(
        chunks=[chunk1, chunk2],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    assert store.count() == 2

    # Query with exact vector matching CHK-1
    results = store.similarity_search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    # First result should be CHK-1 with exact similarity = 1.0
    assert results[0].chunk_id == "CHK-1"
    assert results[0].document_id == "DOC-A"
    assert pytest.approx(results[0].similarity_score, 1e-4) == 1.0
    # Second result should be CHK-2 with orthogonal similarity = 0.0
    assert results[1].chunk_id == "CHK-2"
    assert pytest.approx(results[1].similarity_score, 1e-4) == 0.0

    # Retrieve individual chunk
    retrieved_chunk = store.get_chunk("CHK-1")
    assert retrieved_chunk is not None
    assert retrieved_chunk.chunk_id == "CHK-1"
    assert retrieved_chunk.content == "Kubernetes cluster outage runbook"

    # Non-existent chunk returns None
    assert store.get_chunk("CHK-NONEXISTENT") is None


def test_chroma_vector_store_persistence_and_reopening(tmp_path: Path):
    """Verify vector store persists to disk and can be reopened with data intact."""
    store_dir = tmp_path / "chroma_data"

    # 1. Initialize store, insert data, and close
    store1 = ChromaVectorStore(persist_directory=store_dir, dimension=3)
    chunk1 = _create_sample_chunk("CHK-P1", "DOC-PERSIST", "Persistent operational procedure")
    store1.add_chunks(chunks=[chunk1], embeddings=[[0.0, 0.0, 1.0]])
    assert store1.count() == 1
    assert store_dir.exists()

    # 2. Reopen store from the same directory and verify data recovery
    store2 = ChromaVectorStore(persist_directory=store_dir, dimension=3)
    assert store2.count() == 1
    recovered_chunk = store2.get_chunk("CHK-P1")
    assert recovered_chunk is not None
    assert recovered_chunk.content == "Persistent operational procedure"

    # Verify search continues to function across reloads
    search_res = store2.similarity_search(query_embedding=[0.0, 0.0, 1.0], top_k=1)
    assert len(search_res) == 1
    assert search_res[0].chunk_id == "CHK-P1"
    assert pytest.approx(search_res[0].similarity_score, 1e-4) == 1.0


def test_chroma_vector_store_upsert_and_updates():
    """Verify duplicate chunk IDs update existing content without increasing count."""
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_upsert_updates", dimension=2
    )

    chunk = _create_sample_chunk("CHK-UP", "DOC-1", "Initial draft of SOP")
    store.add_chunks([chunk], embeddings=[[1.0, 0.0]])
    assert store.count() == 1

    # Re-insert with same chunk_id but updated content and vector
    updated_chunk = _create_sample_chunk("CHK-UP", "DOC-1", "Updated revised SOP")
    store.add_chunks([updated_chunk], embeddings=[[0.0, 1.0]])
    assert store.count() == 1  # Count does not duplicate

    # Search confirms updated vector and content
    res = store.similarity_search(query_embedding=[0.0, 1.0], top_k=1)
    assert res[0].chunk_id == "CHK-UP"
    assert res[0].content == "Updated revised SOP"
    assert pytest.approx(res[0].similarity_score, 1e-4) == 1.0


def test_chroma_vector_store_filtering_and_deletion():
    """Verify metadata filtering, document-level deletion, and collection clearing."""
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_filter_delete", dimension=2
    )

    c1 = _create_sample_chunk("C1", "DOC-ALPHA", "Text A")
    c1.metadata["tier"] = "frontend"
    c2 = _create_sample_chunk("C2", "DOC-ALPHA", "Text B")
    c2.metadata["tier"] = "frontend"
    c3 = _create_sample_chunk("C3", "DOC-BETA", "Text C")
    c3.metadata["tier"] = "backend"

    store.add_chunks([c1, c2, c3], [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert store.count() == 3

    # Filter by metadata attribute
    frontend_results = store.similarity_search(
        query_embedding=[1.0, 0.0],
        filter_metadata={"tier": "frontend"},
        top_k=5,
    )
    assert len(frontend_results) == 2
    assert all(r.document_id == "DOC-ALPHA" for r in frontend_results)

    # Filter by document_ids
    beta_results = store.similarity_search(
        query_embedding=[1.0, 0.0],
        document_ids=["DOC-BETA"],
        top_k=5,
    )
    assert len(beta_results) == 1
    assert beta_results[0].document_id == "DOC-BETA"

    # Delete document ALPHA (removes 2 chunks)
    deleted_count = store.delete_document("DOC-ALPHA")
    assert deleted_count == 2
    assert store.count() == 1
    assert store.get_chunk("C1") is None
    assert store.get_chunk("C3") is not None

    # Clear store
    store.clear()
    assert store.count() == 0


def test_chroma_vector_store_dimension_mismatch_and_empty_vectors():
    """Verify dimension validation enforces strict vector bounds."""
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_dim_bounds", dimension=4
    )

    chunk = _create_sample_chunk("C-DIM", "DOC-DIM", "Content")

    # Mismatched embedding on add_chunks raises EmbeddingDimensionMismatchError
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
        store.add_chunks([chunk], embeddings=[[1.0, 0.0]])  # 2 dims != 4
    assert exc_info.value.expected == 4
    assert exc_info.value.actual == 2

    # Mismatched query embedding on similarity_search
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info2:
        store.similarity_search(query_embedding=[1.0, 0.0, 0.0])  # 3 dims != 4
    assert exc_info2.value.expected == 4
    assert exc_info2.value.actual == 3

    # Empty query embedding
    with pytest.raises(VectorStoreError):
        store.similarity_search(query_embedding=[])


def test_chroma_vector_store_error_handling():
    """Verify underlying database exceptions are captured as VectorStoreError."""
    store = ChromaVectorStore(
        persist_directory=":memory:", collection_name="test_err_handling", dimension=2
    )

    # Mismatched chunks and embeddings lengths
    c1 = _create_sample_chunk("C1", "D1", "Text")
    with pytest.raises(VectorStoreError) as exc_info:
        store.add_chunks(chunks=[c1], embeddings=[])
    assert "count" in str(exc_info.value)

    # Simulated client query failure
    with patch.object(store._collection, "query", side_effect=Exception("Database lock error")):
        # Ensure collection is not empty first
        store.add_chunks([c1], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError) as exc_query:
            store.similarity_search(query_embedding=[1.0, 0.0])
        assert "Database lock error" in str(exc_query.value)


def test_get_vector_store_factory(tmp_path: Path):
    """Verify get_vector_store factory resolves backends correctly."""
    store = get_vector_store(
        backend="chroma",
        persist_directory=tmp_path / "test_factory",
        collection_name="factory_collection",
        dimension=32,
    )
    assert isinstance(store, ChromaVectorStore)
    assert store.collection_name == "factory_collection"

    with pytest.raises(VectorStoreError) as exc_info:
        get_vector_store(backend="unsupported_store")
    assert "Unsupported vector store backend" in str(exc_info.value)


def test_chroma_model_and_dimension_validation(tmp_path: Path):
    """Verify ChromaVectorStore enforces embedding model identity and vector dimensionality."""
    store_dir = tmp_path / "model_dim_test"

    # 1. Correct model and dimension: initialization succeeds and stores metadata
    store = ChromaVectorStore(
        persist_directory=store_dir,
        collection_name="col_valid",
        dimension=128,
        embedding_model="nomic-embed-text",
    )
    assert store.dimension == 128
    assert store.embedding_model == "nomic-embed-text"
    del store

    # 2. Persistence and reopening with correct model and dimension succeeds
    store_reopen = ChromaVectorStore(
        persist_directory=store_dir,
        collection_name="col_valid",
        dimension=128,
        embedding_model="nomic-embed-text",
    )
    assert store_reopen.count() == 0
    del store_reopen

    # 3. Incorrect dimension raises EmbeddingDimensionMismatchError
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_dim:
        ChromaVectorStore(
            persist_directory=store_dir,
            collection_name="col_valid",
            dimension=256,
            embedding_model="nomic-embed-text",
        )
    assert exc_dim.value.expected == 256
    assert exc_dim.value.actual == 128

    # 4. Different model with same dimension raises VectorStoreError
    with pytest.raises(VectorStoreError) as exc_model:
        ChromaVectorStore(
            persist_directory=store_dir,
            collection_name="col_valid",
            dimension=128,
            embedding_model="different-embedding-model",
        )
    assert "different-embedding-model" in str(exc_model.value)

    # 5. Existing collection with missing metadata raises VectorStoreError
    import chromadb

    client = chromadb.PersistentClient(path=str(store_dir))
    client.create_collection(name="legacy_no_meta", metadata={"hnsw:space": "cosine"})

    with pytest.raises(VectorStoreError) as exc_meta:
        ChromaVectorStore(
            persist_directory=store_dir,
            collection_name="legacy_no_meta",
            dimension=128,
            embedding_model="nomic-embed-text",
        )
    assert "lacks required embedding metadata" in str(exc_meta.value)


def test_chroma_delete_by_source_path():
    """Verify delete_by_source_path correctly purges chunks while respecting exclude_document_id."""
    store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test_del_by_path",
        dimension=4,
    )

    # Ingest 2 chunks for /docs/runbook.md under DOC-V1
    c1 = _create_sample_chunk("CHK-1", "DOC-V1", "Runbook step 1")
    c1.metadata["source_path"] = "/docs/runbook.md"
    c2 = _create_sample_chunk("CHK-2", "DOC-V1", "Runbook step 2")
    c2.metadata["source_path"] = "/docs/runbook.md"

    # Ingest 1 chunk for /docs/runbook.md under DOC-V2 (replacement)
    c3 = _create_sample_chunk("CHK-3", "DOC-V2", "Runbook updated step")
    c3.metadata["source_path"] = "/docs/runbook.md"

    # Ingest 1 chunk for a DIFFERENT file /other/runbook.md under DOC-OTHER
    c4 = _create_sample_chunk("CHK-4", "DOC-OTHER", "Other runbook")
    c4.metadata["source_path"] = "/other/runbook.md"

    store.add_chunks(
        chunks=[c1, c2, c3, c4],
        embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )
    assert store.count() == 4

    # Delete chunks for /docs/runbook.md excluding DOC-V2
    deleted = store.delete_by_source_path(
        source_path="/docs/runbook.md",
        exclude_document_id="DOC-V2",
    )
    assert deleted == 2  # CHK-1 and CHK-2 deleted
    assert store.count() == 2
    assert store.get_chunk("CHK-1") is None
    assert store.get_chunk("CHK-2") is None
    assert store.get_chunk("CHK-3") is not None
    assert store.get_chunk("CHK-4") is not None

    # Delete all chunks for /other/runbook.md without exclusion
    del_other = store.delete_by_source_path(source_path="/other/runbook.md")
    assert del_other == 1
    assert store.count() == 1
    assert store.get_chunk("CHK-3") is not None

    # Deleting non-existent source path returns 0
    assert store.delete_by_source_path(source_path="/non/existent.md") == 0
