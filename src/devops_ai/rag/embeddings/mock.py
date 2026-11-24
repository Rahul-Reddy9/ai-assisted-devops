# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Deterministic offline mock embedding provider for tests and reproducible benchmarks."""

import hashlib
import math
from typing import List

from devops_ai.core.exceptions import EmbeddingProviderError
from devops_ai.rag.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, offline embedding provider generating reproducible unit-normalized vectors."""

    def __init__(
        self,
        dimension: int = 768,
        model_name: str = "mock-embedding-model",
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension}")
        self._dimension = dimension
        self._model_name = model_name
        self.provider_name = "mock"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def check_health(self) -> bool:
        """Mock provider is always healthy without external services."""
        return True

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """Generate a deterministic, unit-normalized vector for the given text."""
        if not text or not text.strip():
            raise EmbeddingProviderError(
                "Cannot generate embedding for empty or whitespace-only text",
                provider=self.provider_name,
            )

        # Generate base seed from text content hash
        base_seed = hashlib.sha256(text.encode("utf-8")).digest()
        vector: List[float] = []

        # Generate deterministic pseudo-random floats for each dimension
        for dim_idx in range(self._dimension):
            dim_bytes = hashlib.sha256(base_seed + dim_idx.to_bytes(4, byteorder="big")).digest()
            # Map first 4 bytes to float in range [-1.0, 1.0]
            val_int = int.from_bytes(dim_bytes[:4], byteorder="big", signed=True)
            val_float = val_int / (2**31)
            vector.append(val_float)

        # L2-normalize vector to unit length
        squared_sum = sum(x * x for x in vector)
        norm = math.sqrt(squared_sum)
        if norm > 0:
            vector = [x / norm for x in vector]

        self.validate_vector(vector, self._dimension)
        return vector

    def embed_text(self, text: str) -> List[float]:
        """Generate a single deterministic embedding vector."""
        return self._generate_deterministic_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate a batch of deterministic embedding vectors."""
        if not texts:
            return []
        return [self._generate_deterministic_vector(t) for t in texts]
