# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit and mock tests for vector embedding providers and validation routines."""

import math
from unittest.mock import MagicMock, patch
import pytest

from devops_ai.core.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
)
from devops_ai.rag.embeddings import (
    MockEmbeddingProvider,
    OllamaEmbeddingProvider,
    get_embedding_provider,
)


def test_mock_embedding_provider_single_and_batch():
    """Verify MockEmbeddingProvider generates deterministic, unit-normalized vectors."""
    provider = MockEmbeddingProvider(dimension=64, model_name="test-mock")
    assert provider.dimension == 64
    assert provider.model_name == "test-mock"
    assert provider.check_health() is True

    # Single text embedding
    vec1 = provider.embed_text("Kubernetes cluster failover procedures")
    assert len(vec1) == 64
    assert all(math.isfinite(x) for x in vec1)

    # Unit normalization check (L2 norm ~ 1.0)
    norm = math.sqrt(sum(x * x for x in vec1))
    assert pytest.approx(norm, 1e-4) == 1.0

    # Determinism: identical text yields identical vector
    vec2 = provider.embed_text("Kubernetes cluster failover procedures")
    assert vec1 == vec2

    # Different text yields different vector
    vec3 = provider.embed_text("Ingress HTTP 502 Bad Gateway")
    assert vec1 != vec3

    # Batch embedding
    batch = provider.embed_batch(
        ["PodCrashLoopBackOff diagnosis", "OOMKilled container event", "Node DiskPressure"]
    )
    assert len(batch) == 3
    assert all(len(v) == 64 for v in batch)

    # Empty batch returns empty list
    assert provider.embed_batch([]) == []


def test_mock_embedding_provider_validation_and_errors():
    """Verify MockEmbeddingProvider enforces input constraints and dimension bounds."""
    # Invalid dimension bounds
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dimension=0)
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dimension=-10)

    provider = MockEmbeddingProvider(dimension=32)

    # Rejection of empty or whitespace-only inputs
    with pytest.raises(EmbeddingProviderError) as exc_info:
        provider.embed_text("")
    assert "empty or whitespace-only" in str(exc_info.value)

    with pytest.raises(EmbeddingProviderError):
        provider.embed_text("   \n\t  ")

    with pytest.raises(EmbeddingProviderError):
        provider.embed_batch(["valid text", ""])


def test_ollama_embedding_provider_success():
    """Verify OllamaEmbeddingProvider calls ollama.Client.embed and validates output vectors."""
    provider = OllamaEmbeddingProvider(
        model="nomic-embed-text",
        host="http://localhost:11434",
        dimension=4,
    )
    assert provider.dimension == 4
    assert provider.model_name == "nomic-embed-text"
    assert provider.host == "http://localhost:11434"

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    mock_client.embed.return_value = mock_resp

    with patch.object(provider, "_get_client", return_value=mock_client):
        # Single text
        vec = provider.embed_text("Check cluster health")
        assert vec == [0.1, 0.2, 0.3, 0.4]
        mock_client.embed.assert_called_with(model="nomic-embed-text", input="Check cluster health")

        # Batch text
        mock_resp.embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        batch_vecs = provider.embed_batch(["Error in log", "High CPU"])
        assert len(batch_vecs) == 2
        assert batch_vecs[0] == [0.1, 0.2, 0.3, 0.4]
        assert batch_vecs[1] == [0.5, 0.6, 0.7, 0.8]


def test_ollama_embedding_provider_dimension_mismatch():
    """Verify dimension mismatches from provider response raise EmbeddingDimensionMismatchError."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", dimension=768)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings = [[0.1, 0.2, 0.3]]  # 3 dimensions instead of expected 768
    mock_client.embed.return_value = mock_resp

    with patch.object(provider, "_get_client", return_value=mock_client):
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            provider.embed_text("Telemetry payload")
        assert exc_info.value.expected == 768
        assert exc_info.value.actual == 3


def test_ollama_embedding_provider_invalid_elements():
    """Verify non-finite float values (NaN, Inf) from provider are rejected."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", dimension=3)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings = [[0.1, float("nan"), 0.3]]
    mock_client.embed.return_value = mock_resp

    with patch.object(provider, "_get_client", return_value=mock_client):
        with pytest.raises(EmbeddingProviderError) as exc_info:
            provider.embed_text("Corrupted vector")
        assert "non-finite" in str(exc_info.value)

    # Infinity check
    mock_resp.embeddings = [[0.1, float("inf"), 0.3]]
    with patch.object(provider, "_get_client", return_value=mock_client):
        with pytest.raises(EmbeddingProviderError):
            provider.embed_text("Infinite vector")


def test_ollama_embedding_provider_errors_and_secret_scrubbing():
    """Verify Ollama network failures are wrapped in EmbeddingProviderError with secret redaction."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", host="http://localhost:11434")

    mock_client = MagicMock()
    # Simulate connection error exposing a sensitive secret key
    mock_client.embed.side_effect = Exception(
        "Connection refused to http://localhost:11434 with token AIzaSyD3fakeSecretKey1234567890"
    )

    with patch.object(provider, "_get_client", return_value=mock_client):
        with pytest.raises(EmbeddingProviderError) as exc_info:
            provider.embed_text("Query runbook")
        err_msg = str(exc_info.value)
        assert "[REDACTED_API_KEY]" in err_msg
        assert "AIzaSyD3fakeSecretKey1234567890" not in err_msg


def test_ollama_embedding_provider_health_check():
    """Verify check_health queries Ollama tags endpoint and matches models."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text:latest")

    mock_client = MagicMock()
    mock_client.tags.return_value = {
        "models": [{"name": "nomic-embed-text:latest"}, {"name": "llama3.2:1b"}]
    }

    with patch.object(provider, "_get_client", return_value=mock_client):
        assert provider.check_health() is True

    # Failure returns False without crashing
    mock_client.tags.side_effect = Exception("Daemon unavailable")
    with patch.object(provider, "_get_client", return_value=mock_client):
        assert provider.check_health() is False


def test_get_embedding_provider_factory():
    """Verify get_embedding_provider factory instantiates providers correctly."""
    mock_prov = get_embedding_provider("mock", dimension=128)
    assert isinstance(mock_prov, MockEmbeddingProvider)
    assert mock_prov.dimension == 128

    ollama_prov = get_embedding_provider("ollama", model="nomic-embed-text", dimension=768)
    assert isinstance(ollama_prov, OllamaEmbeddingProvider)
    assert ollama_prov.dimension == 768

    with pytest.raises(EmbeddingProviderError) as exc_info:
        get_embedding_provider("unsupported_backend")
    assert "Unsupported embedding provider" in str(exc_info.value)
