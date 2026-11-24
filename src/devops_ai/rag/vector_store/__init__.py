# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Vector storage backends and factory functions."""

from pathlib import Path
from typing import Optional, Union

from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import VectorStoreError
from devops_ai.rag.vector_store.base import BaseVectorStore
from devops_ai.rag.vector_store.chroma import ChromaVectorStore


def get_vector_store(
    backend: Optional[str] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: Optional[str] = None,
    dimension: Optional[int] = None,
    embedding_model: Optional[str] = None,
) -> BaseVectorStore:
    """Resolve and instantiate a vector store backend based on configuration or override."""
    settings = get_settings()
    b_name = (backend or settings.rag_vector_store).strip().lower()

    if b_name == "chroma":
        c_name = collection_name or "devops_ai_rag"
        p_dir = persist_directory or settings.rag_vector_store_path
        dim = dimension or settings.rag_embedding_dimension
        model = embedding_model or settings.rag_embedding_model
        return ChromaVectorStore(
            persist_directory=p_dir,
            collection_name=c_name,
            dimension=dim,
            embedding_model=model,
        )
    else:
        raise VectorStoreError(
            f"Unsupported vector store backend '{b_name}'. Supported backends: ['chroma']",
            store_name=b_name,
        )


__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "get_vector_store",
]
