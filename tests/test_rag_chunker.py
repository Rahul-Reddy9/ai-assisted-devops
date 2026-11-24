# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for SmartChunker structure-aware document splitting and metadata tracking."""

import pytest

from devops_ai.rag.chunker import SmartChunker
from devops_ai.rag.models import DocumentFormat, DocumentMetadata, SourceDocument


def _make_source_doc(content: str, doc_id: str = "DOC-TEST1234") -> SourceDocument:
    meta = DocumentMetadata(
        source_filename="test.md",
        source_format=DocumentFormat.MARKDOWN,
        source_path="/test.md",
        content_hash="hash1234567890abcdef",
        file_size_bytes=len(content.encode("utf-8")),
        character_count=len(content),
    )
    return SourceDocument(document_id=doc_id, raw_content=content, metadata=meta)


def test_chunker_initialization_and_validation():
    """Verify SmartChunker parameters and validation bounds."""
    chunker = SmartChunker(chunk_size=500, chunk_overlap=50)
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 50

    with pytest.raises(ValueError):
        SmartChunker(chunk_size=0)

    with pytest.raises(ValueError):
        SmartChunker(chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError):
        SmartChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError):
        SmartChunker(chunk_size=500, chunk_overlap=600)


def test_chunker_short_document_single_chunk():
    """Verify short documents under chunk_size generate exactly one chunk."""
    chunker = SmartChunker(chunk_size=1000, chunk_overlap=100)
    doc = _make_source_doc("Short operational note on database maintenance window.")

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.total_chunks == 1
    assert chunk.document_id == "DOC-TEST1234"
    assert "Short operational note" in chunk.content
    assert chunk.metadata["source_filename"] == "test.md"


def test_chunker_preserves_code_blocks():
    """Verify markdown code blocks are kept intact as atomic units when fitting within chunk size."""
    chunker = SmartChunker(chunk_size=600, chunk_overlap=50)
    content = (
        "# Deployment Checklist\n\n"
        "Ensure all deployment preconditions are satisfied:\n\n"
        "```yaml\n"
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        "  name: nginx-deployment\n"
        "spec:\n"
        "  replicas: 3\n"
        "```\n\n"
        "Verify pods are healthy after rollout completes."
    )
    doc = _make_source_doc(content)
    chunks = chunker.chunk_document(doc)

    # Ensure code block is contained in full inside a chunk
    found_code_chunk = False
    for c in chunks:
        if "kind: Deployment" in c.content:
            assert "```yaml" in c.content
            assert "replicas: 3" in c.content
            assert c.metadata["has_code_block"] is True
            found_code_chunk = True
    assert found_code_chunk is True


def test_chunker_multi_chunk_with_overlap():
    """Verify chunk generation splits across size limits and provides overlap."""
    chunker = SmartChunker(chunk_size=300, chunk_overlap=60)
    # Generate multi-paragraph text exceeding 300 chars
    paragraphs = [
        f"Paragraph {i}: Site reliability engineering practices require continuous monitoring of golden signals "
        f"including latency, traffic, errors, and saturation across all microservices."
        for i in range(1, 6)
    ]
    doc = _make_source_doc("\n\n".join(paragraphs))
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    # Check deterministic indexing and total_chunks synchronization
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.total_chunks == len(chunks)
        assert c.chunk_id == f"CHK-DOC-TEST1234-{idx:04d}"

    # Verify overlap exists between consecutive chunks
    c0 = chunks[0].content
    c1 = chunks[1].content
    # The start of chunk 1 should contain trailing words from chunk 0
    overlap_found = any(word in c1 for word in c0.split()[-4:])
    assert overlap_found is True


def test_chunker_oversized_block_safe_handling():
    """Verify oversized blocks exceeding chunk_size are split safely without errors."""
    chunker = SmartChunker(chunk_size=200, chunk_overlap=30)
    huge_paragraph = (
        "This is an extremely long unbroken paragraph explaining disaster recovery procedures in exhaustive detail. "
        "It contains multiple sentences detailing replication topology, database failover steps, DNS TTL propagation delays, "
        "load balancer health probe adjustments, and cache invalidation protocols across multiple cloud regions."
    )
    doc = _make_source_doc(huge_paragraph)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.content) <= 250  # Bounded within approximate chunk size
        assert len(c.content) > 0


def test_chunker_empty_document():
    """Verify chunker returns empty list for empty content."""
    chunker = SmartChunker()
    doc = _make_source_doc("")
    chunks = chunker.chunk_document(doc)
    assert chunks == []
