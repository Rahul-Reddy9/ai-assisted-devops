# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Abstract base class and contract for persistent vector stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from devops_ai.rag.models import DocumentChunk, RetrievalResult


class BaseVectorStore(ABC):
    """Abstract base class establishing the contract for dense vector persistence and search."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Persist or update document chunks with their dense vector embeddings."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Execute semantic similarity search against indexed vector chunks."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks associated with the specified document identifier.

        Returns:
            Number of chunks removed.
        """

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by its deterministic identifier."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of indexed vector chunks."""

    @abstractmethod
    def clear(self) -> None:
        """Purge all chunks and vectors from the store."""
