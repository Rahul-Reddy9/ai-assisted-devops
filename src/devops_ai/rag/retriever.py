# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Semantic retrieval engine coordinating embedding generation and vector similarity search."""

from typing import Any, Dict, List, Optional

from devops_ai.core.config import Settings, get_settings
from devops_ai.core.exceptions import (
    EmbeddingProviderError,
    RetrievalError,
    VectorStoreError,
)
from devops_ai.core.logging import get_logger, scrub_secrets
from devops_ai.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from devops_ai.rag.models import RetrievalResult
from devops_ai.rag.vector_store import BaseVectorStore, get_vector_store

logger = get_logger("rag.retriever")


class RAGRetriever:
    """Semantic retrieval coordinator querying vector stores via dense embeddings."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()
        self.default_top_k = self.settings.rag_retrieval_top_k
        self.default_threshold = self.settings.rag_similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
        deduplicate_content: bool = True,
    ) -> List[RetrievalResult]:
        """Execute semantic similarity search for a query and return ranked results.

        Args:
            query: Natural language diagnostic inquiry or question.
            top_k: Maximum number of retrieved chunks (defaults to settings.rag_retrieval_top_k).
            score_threshold: Minimum cosine similarity threshold (defaults to settings.rag_similarity_threshold).
            filter_metadata: Optional metadata filter key-values.
            document_ids: Optional list of document IDs to restrict search scope.
            deduplicate_content: When True, filters out duplicate chunks with identical content hashes.

        Returns:
            List of scored and ranked RetrievalResult objects.
        """
        if not query or not query.strip():
            logger.debug(
                "Empty or whitespace-only query provided to retriever; returning empty results"
            )
            return []

        clean_query = query.strip()
        effective_k = top_k if top_k is not None else self.default_top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self.default_threshold
        )

        if effective_k <= 0:
            return []

        try:
            # 1. Generate dense query embedding
            query_embedding = self.embedding_provider.embed_text(clean_query)

            # 2. Search persistent vector store
            raw_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=effective_k,
                score_threshold=effective_threshold,
                filter_metadata=filter_metadata,
                document_ids=document_ids,
            )
        except (EmbeddingProviderError, VectorStoreError):
            raise
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            logger.error(f"Unexpected retrieval failure for query '{clean_query}': {sanitized}")
            raise RetrievalError(f"Failed to execute semantic retrieval: {sanitized}") from e

        # 3. Optional content deduplication (collapse duplicate chunks)
        filtered_results: List[RetrievalResult] = []
        seen_hashes = set()
        seen_ids = set()

        for item in raw_results:
            if item.chunk_id in seen_ids:
                continue
            seen_ids.add(item.chunk_id)

            if deduplicate_content:
                c_hash = item.metadata.get("content_hash", "")
                if c_hash:
                    if c_hash in seen_hashes:
                        continue
                    seen_hashes.add(c_hash)

            filtered_results.append(item)

        # 4. Deterministic ordering: similarity descending, chunk_id ascending as tie-breaker
        filtered_results.sort(key=lambda r: (-r.similarity_score, r.chunk_id))
        return filtered_results[:effective_k]
