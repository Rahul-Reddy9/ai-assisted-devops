# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""RAG document ingestion, normalization, and chunking subsystem."""

from devops_ai.core.exceptions import (
    DocumentPageLimitError,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    RetrievalError,
    VectorStoreError,
)
from devops_ai.rag.chunker import SmartChunker
from devops_ai.rag.embeddings import (
    BaseEmbeddingProvider,
    MockEmbeddingProvider,
    OllamaEmbeddingProvider,
    get_embedding_provider,
)
from devops_ai.rag.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
    EvaluationReport,
    RAGEvaluator,
)
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
    RetrievalResult,
    SourceDocument,
)
from devops_ai.rag.pipeline import DocumentIngestionPipeline
from devops_ai.rag.preprocessor import DocumentPreprocessor
from devops_ai.rag.retriever import RAGRetriever
from devops_ai.rag.vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
    get_vector_store,
)

__all__ = [
    "BaseEmbeddingProvider",
    "BaseExtractor",
    "BaseVectorStore",
    "ChromaVectorStore",
    "DocumentChunk",
    "DocumentFormat",
    "DocumentIngestionPipeline",
    "DocumentMetadata",
    "DocumentPageLimitError",
    "DocumentPreprocessor",
    "EmbeddingDimensionMismatchError",
    "EmbeddingProviderError",
    "EvaluationDataset",
    "EvaluationQuery",
    "EvaluationReport",
    "IngestionResult",
    "IngestionStatus",
    "JSONExtractor",
    "MarkdownExtractor",
    "MockEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "PDFExtractor",
    "PlainTextExtractor",
    "RAGEvaluator",
    "RAGRetriever",
    "RetrievalError",
    "RetrievalResult",
    "SmartChunker",
    "SourceDocument",
    "VectorStoreError",
    "YAMLExtractor",
    "get_embedding_provider",
    "get_extractor",
    "get_vector_store",
]
