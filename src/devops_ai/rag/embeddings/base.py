# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Abstract base class and contract for dense vector embedding providers."""

from abc import ABC, abstractmethod
import math
from typing import List

from devops_ai.core.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
)


class BaseEmbeddingProvider(ABC):
    """Abstract base class establishing the contract for vector embedding generators."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single text string."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of text strings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality produced by this embedding model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active embedding model name or identifier."""

    @abstractmethod
    def check_health(self) -> bool:
        """Verify provider availability and model readiness."""

    def validate_vector(self, vector: List[float], expected_dim: int) -> None:
        """Validate that a generated vector meets dimensionality and finite numerical bounds.

        Raises:
            EmbeddingDimensionMismatchError: If the vector length does not match expected_dim.
            EmbeddingProviderError: If the vector is empty or contains non-finite values (NaN/Inf).
        """
        if not vector:
            raise EmbeddingProviderError(
                "Embedding vector is empty",
                provider=getattr(self, "provider_name", "unknown"),
            )

        actual_dim = len(vector)
        if actual_dim != expected_dim:
            raise EmbeddingDimensionMismatchError(
                f"Generated vector dimension {actual_dim} does not match expected {expected_dim}",
                expected=expected_dim,
                actual=actual_dim,
            )

        for idx, val in enumerate(vector):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise EmbeddingProviderError(
                    f"Vector contains invalid or non-finite element at index {idx}: {val}",
                    provider=getattr(self, "provider_name", "unknown"),
                )
