# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Ollama local embedding provider implementation using nomic-embed-text."""

from typing import List, Optional

from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
)
from devops_ai.core.logging import get_logger, scrub_secrets
from devops_ai.rag.embeddings.base import BaseEmbeddingProvider

logger = get_logger("rag.embeddings.ollama")


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Local dense vector embedding provider utilizing the Ollama daemon."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        dimension: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.rag_embedding_model
        self._host = (host or settings.ollama_host).rstrip("/")
        self._dimension = dimension if dimension is not None else settings.rag_embedding_dimension
        self._timeout = timeout if timeout is not None else settings.llm_timeout_seconds
        self.provider_name = "ollama"

        if self._dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def host(self) -> str:
        return self._host

    def _get_client(self):
        try:
            import ollama

            return ollama.Client(host=self._host, timeout=self._timeout)
        except ImportError:
            raise EmbeddingProviderError(
                "Ollama package is not installed. Install via: pip install 'devops-ai[ollama]'",
                provider=self.provider_name,
            ) from None

    def check_health(self) -> bool:
        """Verify Ollama service reachability and model availability."""
        try:
            client = self._get_client()
            tags_response = client.tags()
            models = [
                m.get("name", "") if isinstance(m, dict) else getattr(m, "model", "")
                for m in tags_response.get("models", [])
            ]
            # Match exact model name or prefix before tag
            matched = any(
                self._model == m or self._model.split(":")[0] == m.split(":")[0] for m in models
            )
            if not matched:
                logger.warning(
                    f"Model '{self._model}' not found in Ollama tags. Available: {models}"
                )
            return True
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            logger.debug(f"Ollama health check probe failed at '{self._host}': {sanitized}")
            return False

    def embed_text(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single text string."""
        if not text or not text.strip():
            raise EmbeddingProviderError(
                "Cannot generate embedding for empty or whitespace-only text",
                provider=self.provider_name,
            )

        client = self._get_client()
        try:
            response = client.embed(model=self._model, input=text)
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")

            if not embeddings or not embeddings[0]:
                raise EmbeddingProviderError(
                    f"Ollama returned empty embedding response for model '{self._model}'",
                    provider=self.provider_name,
                )

            vector = list(embeddings[0])
            self.validate_vector(vector, self._dimension)
            return vector
        except (EmbeddingDimensionMismatchError, EmbeddingProviderError):
            raise
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            logger.error(f"Ollama embedding failure on '{self._model}': {sanitized}")
            raise EmbeddingProviderError(
                f"Ollama embedding request failed at '{self._host}' with model '{self._model}': {sanitized}",
                provider=self.provider_name,
            ) from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of text strings."""
        if not texts:
            return []

        for idx, t in enumerate(texts):
            if not t or not t.strip():
                raise EmbeddingProviderError(
                    f"Cannot generate embedding: item at index {idx} is empty or whitespace-only",
                    provider=self.provider_name,
                )

        client = self._get_client()
        try:
            response = client.embed(model=self._model, input=texts)
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")

            if not embeddings:
                raise EmbeddingProviderError(
                    f"Ollama returned empty batch embedding response for model '{self._model}'",
                    provider=self.provider_name,
                )

            vectors: List[List[float]] = []
            for idx, vec in enumerate(embeddings):
                v_list = list(vec)
                self.validate_vector(v_list, self._dimension)
                vectors.append(v_list)

            return vectors
        except (EmbeddingDimensionMismatchError, EmbeddingProviderError):
            raise
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            logger.error(f"Ollama batch embedding failure on '{self._model}': {sanitized}")
            raise EmbeddingProviderError(
                f"Ollama batch embedding request failed at '{self._host}' with model '{self._model}': {sanitized}",
                provider=self.provider_name,
            ) from e
