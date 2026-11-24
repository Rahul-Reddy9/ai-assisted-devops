# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""RAG document ingestion, normalization, and chunking subsystem."""

from devops_ai.core.exceptions import DocumentPageLimitError
from devops_ai.rag.chunker import SmartChunker
from devops_ai.rag.extractors import (
    BaseExtractor,
    JSONExtractor,
    MarkdownExtractor,
    PDFExtractor,
    PlainTextExtractor,
    YAMLExtractor,
    get_extractor,
)
from devops_ai.rag.models import (
    DocumentChunk,
    DocumentFormat,
    DocumentMetadata,
    IngestionResult,
    IngestionStatus,
    SourceDocument,
)
from devops_ai.rag.pipeline import DocumentIngestionPipeline
from devops_ai.rag.preprocessor import DocumentPreprocessor

__all__ = [
    "BaseExtractor",
    "DocumentChunk",
    "DocumentFormat",
    "DocumentIngestionPipeline",
    "DocumentMetadata",
    "DocumentPageLimitError",
    "DocumentPreprocessor",
    "IngestionResult",
    "IngestionStatus",
    "JSONExtractor",
    "MarkdownExtractor",
    "PDFExtractor",
    "PlainTextExtractor",
    "SmartChunker",
    "SourceDocument",
    "YAMLExtractor",
    "get_extractor",
]
