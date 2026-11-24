# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Core configuration, exceptions, and logging facilities."""

from devops_ai.core.config import Settings, get_settings
from devops_ai.core.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentLimitExceededError,
    AgentNotFoundError,
    AgentSecurityError,
    ChunkingError,
    ConfigurationError,
    DevOpsAIError,
    DocumentFormatError,
    DocumentIngestionError,
    DocumentPageLimitError,
    DocumentSecurityError,
    DocumentSizeLimitError,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    LLMProviderError,
    ObservabilityError,
    RAGError,
    RetrievalError,
    VectorStoreError,
)
from devops_ai.core.logging import get_logger, setup_logging

__all__ = [
    "AgentError",
    "AgentExecutionError",
    "AgentLimitExceededError",
    "AgentNotFoundError",
    "AgentSecurityError",
    "ChunkingError",
    "DocumentFormatError",
    "DocumentIngestionError",
    "DocumentPageLimitError",
    "DocumentSecurityError",
    "DocumentSizeLimitError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingProviderError",
    "RAGError",
    "RetrievalError",
    "VectorStoreError",
    "Settings",
    "get_settings",
    "DevOpsAIError",
    "ConfigurationError",
    "LLMProviderError",
    "ObservabilityError",
    "get_logger",
    "setup_logging",
]
