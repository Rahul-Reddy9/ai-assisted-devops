# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Data models for RAG document ingestion, metadata management, and document chunking."""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from devops_ai.core.exceptions import DocumentFormatError


class DocumentFormat(str, Enum):
    """Supported document formats for RAG ingestion."""

    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"

    @classmethod
    def from_extension(cls, extension: str) -> "DocumentFormat":
        """Map file extension to DocumentFormat or raise DocumentFormatError."""
        ext = extension.strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        mapping = {
            ".md": cls.MARKDOWN,
            ".markdown": cls.MARKDOWN,
            ".txt": cls.TEXT,
            ".text": cls.TEXT,
            ".pdf": cls.PDF,
            ".json": cls.JSON,
            ".yaml": cls.YAML,
            ".yml": cls.YAML,
        }
        if ext in mapping:
            return mapping[ext]
        raise DocumentFormatError(
            f"Unsupported file extension '{extension}'. Supported extensions: {list(mapping.keys())}",
            format_type=extension,
        )


class IngestionStatus(str, Enum):
    """Status outcomes for document ingestion."""

    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class DocumentMetadata(BaseModel):
    """Metadata describing a source document and its provenance."""

    model_config = ConfigDict(extra="forbid")

    source_filename: str = Field(min_length=1, description="Basename of the source document")
    source_format: DocumentFormat = Field(description="Format classification of the document")
    source_path: str = Field(min_length=1, description="Source file path or logical URI")
    content_hash: str = Field(min_length=16, description="SHA-256 hash of normalized content")
    file_size_bytes: int = Field(ge=0, description="Raw file size in bytes")
    character_count: int = Field(ge=0, description="Character count of extracted content")
    ingestion_timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC timestamp of ingestion",
    )
    custom_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted or user-supplied metadata attributes (title, section, tags)",
    )


class DocumentChunk(BaseModel):
    """A discrete, indexed segment of a source document prepared for embedding."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, description="Deterministic unique identifier for the chunk")
    document_id: str = Field(min_length=1, description="Identifier of the parent source document")
    chunk_index: int = Field(ge=0, description="0-based index of this chunk within the document")
    total_chunks: int = Field(ge=1, description="Total number of chunks produced for the document")
    content: str = Field(min_length=1, description="Extracted and normalized chunk text")
    character_count: int = Field(ge=1, description="Length of chunk content in characters")
    content_hash: str = Field(min_length=16, description="SHA-256 hash of chunk content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-level metadata including source provenance and boundaries",
    )


class SourceDocument(BaseModel):
    """In-memory representation of an ingested document before chunking."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        default_factory=lambda: f"DOC-{uuid.uuid4().hex[:12].upper()}",
        description="Unique document identifier",
    )
    raw_content: str = Field(description="Extracted normalized text content of the document")
    metadata: DocumentMetadata = Field(description="Document metadata and provenance")


class IngestionResult(BaseModel):
    """Outcome of document ingestion and chunking."""

    model_config = ConfigDict(extra="forbid")

    document_id: Optional[str] = Field(default=None, description="Document identifier if ingested")
    source_path: str = Field(description="Source file path or identifier")
    status: IngestionStatus = Field(description="Final status of the ingestion operation")
    chunks: List[DocumentChunk] = Field(
        default_factory=list, description="Generated document chunks"
    )
    total_chunks: int = Field(default=0, ge=0, description="Count of chunks generated")
    content_hash: Optional[str] = Field(
        default=None, description="SHA-256 hash of document content"
    )
    error: Optional[str] = Field(default=None, description="Error message if ingestion failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic or ingestion metadata",
    )
