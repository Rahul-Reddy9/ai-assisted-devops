# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for RAG data models, serialization, and schema constraints."""

import datetime
import pytest
from pydantic import ValidationError

from devops_ai.core.exceptions import DocumentFormatError
from devops_ai.rag.models import (
    DocumentChunk,
    DocumentFormat,
    DocumentMetadata,
    IngestionResult,
    IngestionStatus,
    SourceDocument,
)


def test_document_format_enum_and_extension_mapping():
    """Verify supported formats and file extension mapping."""
    assert DocumentFormat.from_extension(".md") == DocumentFormat.MARKDOWN
    assert DocumentFormat.from_extension("md") == DocumentFormat.MARKDOWN
    assert DocumentFormat.from_extension(".markdown") == DocumentFormat.MARKDOWN
    assert DocumentFormat.from_extension(".txt") == DocumentFormat.TEXT
    assert DocumentFormat.from_extension(".pdf") == DocumentFormat.PDF
    assert DocumentFormat.from_extension(".json") == DocumentFormat.JSON
    assert DocumentFormat.from_extension(".yaml") == DocumentFormat.YAML
    assert DocumentFormat.from_extension(".yml") == DocumentFormat.YAML

    with pytest.raises(DocumentFormatError) as exc_info:
        DocumentFormat.from_extension(".docx")
    assert "Unsupported file extension '.docx'" in str(exc_info.value)


def test_document_metadata_model():
    """Verify DocumentMetadata fields and validation."""
    meta = DocumentMetadata(
        source_filename="runbook.md",
        source_format=DocumentFormat.MARKDOWN,
        source_path="/docs/sre/runbook.md",
        content_hash="a1b2c3d4e5f678901234567890abcdef",
        file_size_bytes=1024,
        character_count=850,
        custom_attributes={"service": "checkout", "author": "sre-team"},
    )
    assert meta.source_filename == "runbook.md"
    assert meta.source_format == DocumentFormat.MARKDOWN
    assert meta.file_size_bytes == 1024
    assert meta.custom_attributes["service"] == "checkout"
    assert isinstance(meta.ingestion_timestamp, datetime.datetime)

    # Verify extra forbidden
    with pytest.raises(ValidationError):
        DocumentMetadata(
            source_filename="runbook.md",
            source_format=DocumentFormat.MARKDOWN,
            source_path="/docs/sre/runbook.md",
            content_hash="a1b2c3d4e5f678901234567890abcdef",
            file_size_bytes=1024,
            character_count=850,
            unauthorized_field="forbidden",
        )


def test_document_chunk_model():
    """Verify DocumentChunk constraints and properties."""
    chunk = DocumentChunk(
        chunk_id="CHK-DOC12345678-0000",
        document_id="DOC12345678",
        chunk_index=0,
        total_chunks=3,
        content="This is the first chunk of operational runbook documentation.",
        character_count=61,
        content_hash="c1d2e3f4a5b678901234567890abcdef",
        metadata={"has_code_block": False},
    )
    assert chunk.chunk_id == "CHK-DOC12345678-0000"
    assert chunk.chunk_index == 0
    assert chunk.total_chunks == 3
    assert chunk.character_count == 61

    # Empty content rejected
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="CHK-1",
            document_id="DOC1",
            chunk_index=0,
            total_chunks=1,
            content="",
            character_count=0,
            content_hash="hash",
        )


def test_source_document_and_ingestion_result():
    """Verify SourceDocument and IngestionResult serialization roundtrip."""
    meta = DocumentMetadata(
        source_filename="notes.txt",
        source_format=DocumentFormat.TEXT,
        source_path="/notes.txt",
        content_hash="hash1234567890abcdef",
        file_size_bytes=100,
        character_count=90,
    )
    source_doc = SourceDocument(
        document_id="DOC-TEST001",
        raw_content="Cluster maintenance procedures.",
        metadata=meta,
    )
    assert source_doc.document_id == "DOC-TEST001"

    res = IngestionResult(
        document_id="DOC-TEST001",
        source_path="/notes.txt",
        status=IngestionStatus.SUCCESS,
        total_chunks=1,
        content_hash="hash1234567890abcdef",
    )
    json_data = res.model_dump_json()
    assert "DOC-TEST001" in json_data
    assert "SUCCESS" in json_data
