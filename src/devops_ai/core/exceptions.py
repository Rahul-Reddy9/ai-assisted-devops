# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Domain exceptions for the DevOps AI framework."""


class DevOpsAIError(Exception):
    """Base exception for all DevOps AI errors."""

    def __init__(self, message: str, code: str = "DEVOPS_AI_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigurationError(DevOpsAIError):
    """Raised when configuration values are missing, invalid, or malformed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIG_ERROR")


class LLMProviderError(DevOpsAIError):
    """Raised when an LLM provider fails to generate or connect."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(f"Provider '{provider}' error: {message}", code="LLM_PROVIDER_ERROR")
        self.provider = provider


class ObservabilityError(DevOpsAIError):
    """Raised when log parsing or anomaly detection fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="OBSERVABILITY_ERROR")


class IncidentError(DevOpsAIError):
    """Raised when an incident operation fails."""

    def __init__(self, message: str, code: str = "INCIDENT_ERROR") -> None:
        super().__init__(message, code=code)


class IncidentNotFoundError(IncidentError):
    """Raised when a requested incident cannot be located."""

    def __init__(self, incident_id: str) -> None:
        super().__init__(f"Incident not found: '{incident_id}'", code="INCIDENT_NOT_FOUND")
        self.incident_id = incident_id


class ConcurrencyConflictError(IncidentError):
    """Raised when an optimistic concurrency version conflict occurs during repository update."""

    def __init__(self, incident_id: str, expected: int, actual: int) -> None:
        super().__init__(
            f"Concurrency conflict for incident '{incident_id}': expected version {expected}, found {actual}",
            code="CONCURRENCY_CONFLICT",
        )
        self.incident_id = incident_id
        self.expected = expected
        self.actual = actual


class RepositoryLockError(IncidentError):
    """Raised when the repository file lock cannot be acquired or broken safely."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="REPOSITORY_LOCK_ERROR")


# =====================================================================
# AI Agent Architecture Exceptions (Phase 5)
# =====================================================================


class AgentError(DevOpsAIError):
    """Base exception for all AI agent errors."""

    def __init__(self, message: str, code: str = "AGENT_ERROR") -> None:
        super().__init__(message, code=code)


class AgentExecutionError(AgentError):
    """Raised when an agent execution fails during planning or task execution."""

    def __init__(self, message: str, agent_id: str = "unknown") -> None:
        super().__init__(
            f"Agent '{agent_id}' execution failed: {message}",
            code="AGENT_EXECUTION_ERROR",
        )
        self.agent_id = agent_id


class AgentLimitExceededError(AgentError):
    """Raised when an agent exceeds execution limits (timeout, iterations, token budget)."""

    def __init__(self, message: str, limit_type: str = "general") -> None:
        super().__init__(
            f"Agent limit exceeded ({limit_type}): {message}",
            code="AGENT_LIMIT_EXCEEDED",
        )
        self.limit_type = limit_type


class AgentSecurityError(AgentError):
    """Raised when an agent attempts an unauthorized mutation or violates safety boundaries."""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"Agent safety boundary violation: {message}",
            code="AGENT_SECURITY_ERROR",
        )


class AgentNotFoundError(AgentError):
    """Raised when a requested agent cannot be found in the registry."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(
            f"Agent not found in registry: '{agent_name}'",
            code="AGENT_NOT_FOUND",
        )
        self.agent_name = agent_name
