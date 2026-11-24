# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Domain exceptions for the DevOps AI framework."""

from devops_ai.core.exceptions import (
    ConfigurationError,
    DevOpsAIError,
    DocumentPageLimitError,
    DocumentSizeLimitError,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    LLMProviderError,
    ObservabilityError,
    RAGError,
    RetrievalError,
    VectorStoreError,
)


def test_base_exception():
    err = DevOpsAIError("Something went wrong", code="CUSTOM_CODE")
    assert str(err) == "[CUSTOM_CODE] Something went wrong"
    assert err.code == "CUSTOM_CODE"


def test_configuration_error():
    err = ConfigurationError("Missing API key")
    assert isinstance(err, DevOpsAIError)
    assert err.code == "CONFIG_ERROR"
    assert "[CONFIG_ERROR] Missing API key" in str(err)


def test_llm_provider_error():
    err = LLMProviderError("Connection refused", provider="ollama")
    assert isinstance(err, DevOpsAIError)
    assert err.provider == "ollama"
    assert err.code == "LLM_PROVIDER_ERROR"
    assert "Provider 'ollama' error: Connection refused" in str(err)


def test_observability_error():
    err = ObservabilityError("Failed to parse log line")
    assert isinstance(err, DevOpsAIError)
    assert err.code == "OBSERVABILITY_ERROR"


def test_document_page_limit_error_formatting_and_inheritance():
    """Verify DocumentPageLimitError message formatting, attributes, and inheritance."""
    err = DocumentPageLimitError(
        "PDF document has excessive pagination",
        page_count=12,
        limit=5,
    )

    # 1. Inheritance checks
    assert isinstance(err, DocumentPageLimitError)
    assert isinstance(err, DocumentSizeLimitError)
    assert isinstance(err, RAGError)
    assert isinstance(err, DevOpsAIError)
    assert issubclass(DocumentPageLimitError, DocumentSizeLimitError)
    assert issubclass(DocumentPageLimitError, RAGError)

    # 2. Attribute checks
    assert err.page_count == 12
    assert err.limit == 5
    assert err.size == 12
    assert err.code == "DOCUMENT_SIZE_LIMIT_ERROR"

    # 3. Message formatting checks
    error_str = str(err)
    assert "[DOCUMENT_SIZE_LIMIT_ERROR]" in error_str
    assert "Document page limit exceeded (12 pages > 5 pages limit)" in error_str
    assert "PDF document has excessive pagination" in error_str

    # 4. Absence of misleading byte-based prefixes
    assert "bytes" not in error_str
    assert "Document size limit exceeded" not in error_str

    # 5. Backward compatibility in exception handling
    caught = False
    try:
        raise err
    except DocumentSizeLimitError as caught_err:
        caught = True
        assert caught_err.page_count == 12
        assert caught_err.limit == 5
        assert caught_err.size == 12
    assert caught is True


def test_document_size_limit_error_remains_unaffected():
    """Verify standard DocumentSizeLimitError retains byte-based prefix formatting."""
    err = DocumentSizeLimitError("Raw payload exceeds threshold", size=2048, limit=1024)
    assert isinstance(err, DocumentSizeLimitError)
    assert isinstance(err, RAGError)
    assert err.size == 2048
    assert err.limit == 1024
    assert err.code == "DOCUMENT_SIZE_LIMIT_ERROR"

    error_str = str(err)
    assert "[DOCUMENT_SIZE_LIMIT_ERROR]" in error_str
    assert "Document size limit exceeded (2048 bytes > 1024 bytes limit)" in error_str
    assert "Raw payload exceeds threshold" in error_str


def test_rag_vector_exceptions_hierarchy_and_formatting():
    """Verify VectorStoreError, EmbeddingProviderError, and RetrievalError behavior."""
    # VectorStoreError
    vs_err = VectorStoreError("Connection to ChromaDB lost", store_name="chroma")
    assert isinstance(vs_err, RAGError)
    assert isinstance(vs_err, DevOpsAIError)
    assert vs_err.code == "VECTOR_STORE_ERROR"
    assert vs_err.store_name == "chroma"
    assert "Vector store 'chroma' error: Connection to ChromaDB lost" in str(vs_err)

    # EmbeddingDimensionMismatchError
    dim_err = EmbeddingDimensionMismatchError("Vector rejected", expected=768, actual=384)
    assert isinstance(dim_err, VectorStoreError)
    assert isinstance(dim_err, RAGError)
    assert dim_err.code == "EMBEDDING_DIMENSION_MISMATCH"
    assert dim_err.expected == 768
    assert dim_err.actual == 384
    assert "expected 768, got 384" in str(dim_err)

    # EmbeddingProviderError
    emb_err = EmbeddingProviderError("Ollama daemon timeout", provider="ollama")
    assert isinstance(emb_err, RAGError)
    assert emb_err.code == "EMBEDDING_PROVIDER_ERROR"
    assert emb_err.provider == "ollama"
    assert "Embedding provider 'ollama' error: Ollama daemon timeout" in str(emb_err)

    # RetrievalError
    ret_err = RetrievalError("Query index syntax malformed")
    assert isinstance(ret_err, RAGError)
    assert ret_err.code == "RETRIEVAL_ERROR"
    assert "Semantic retrieval error: Query index syntax malformed" in str(ret_err)
