# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Vector embedding providers and factory functions."""

from typing import Optional

from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import EmbeddingProviderError
from devops_ai.rag.embeddings.base import BaseEmbeddingProvider
from devops_ai.rag.embeddings.mock import MockEmbeddingProvider
from devops_ai.rag.embeddings.ollama import OllamaEmbeddingProvider


def get_embedding_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    dimension: Optional[int] = None,
    host: Optional[str] = None,
) -> BaseEmbeddingProvider:
    """Resolve and instantiate an embedding provider based on configuration or override."""
    settings = get_settings()
    p_name = (provider_name or settings.rag_embedding_provider).strip().lower()

    if p_name == "ollama":
        return OllamaEmbeddingProvider(model=model, host=host, dimension=dimension)
    elif p_name == "mock":
        dim = dimension if dimension is not None else settings.rag_embedding_dimension
        m_name = model or "mock-embedding-model"
        return MockEmbeddingProvider(dimension=dim, model_name=m_name)
    else:
        raise EmbeddingProviderError(
            f"Unsupported embedding provider '{p_name}'. Supported: ['ollama', 'mock']",
            provider=p_name,
        )


__all__ = [
    "BaseEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "MockEmbeddingProvider",
    "get_embedding_provider",
]
