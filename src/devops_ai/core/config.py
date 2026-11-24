# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Type-safe configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import List, Literal, Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for DevOps AI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "testing", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # LLM Provider Configuration
    default_llm_provider: Literal["ollama", "gemini"] = Field(
        default="ollama", description="Default active LLM provider"
    )
    enable_llm_fallback: bool = Field(
        default=False, description="Enable automatic fallback to alternate provider on failure"
    )
    llm_timeout_seconds: float = Field(
        default=60.0, ge=1.0, le=600.0, description="Provider timeout in seconds"
    )

    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_model: str = Field(default="llama3.2:1b", description="Default local LLM model")

    # Hosted LLM settings (Google Gemini)
    google_api_key: Optional[SecretStr] = Field(
        default=None, description="Google Gemini API Key (SecretStr)"
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash", description="Configurable Gemini model identifier"
    )

    # Observability & Metrics Configuration
    log_format: Literal["rich", "json"] = "rich"
    metrics_host: str = Field(
        default="127.0.0.1", description="Prometheus metrics server bind address"
    )
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Prometheus metrics server port"
    )
    health_check_timeout_seconds: float = Field(
        default=5.0, ge=0.5, le=60.0, description="Health probe timeout in seconds"
    )

    # Observability defaults (Phase 1 preserved)
    anomaly_contamination: float = Field(default=0.1, ge=0.01, le=0.5)
    burst_threshold: int = Field(default=3, ge=1)
    burst_window_seconds: int = Field(default=30, ge=5)

    # Incident Management & Analysis Configuration (Phase 4)
    incident_storage_dir: str = Field(
        default=".devops_ai/incidents",
        description="Local directory for incident repository persistence",
    )
    incident_dedup_cooldown_seconds: int = Field(
        default=300,
        ge=10,
        le=86400,
        description="Deduplication cooldown window in seconds for identical alert fingerprints",
    )
    incident_reopen_window_seconds: int = Field(
        default=3600,
        ge=60,
        le=604800,
        description="Window in seconds during which recurring alerts will reopen a resolved incident",
    )
    incident_correlation_window_seconds: int = Field(
        default=300,
        ge=10,
        le=86400,
        description="Temporal correlation sliding window in seconds",
    )
    incident_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Retention period in days for closed incidents before archiving/pruning",
    )
    incident_max_records: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum active/persisted incidents retained in primary store",
    )
    rca_confidence_threshold: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="Confidence threshold below which Tier 2 AI investigation is triggered",
    )
    rca_auto_tier2: bool = Field(
        default=False,
        description="Automatically escalate to Tier 2 AI investigation when confidence is below threshold",
    )
    rca_ai_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds for Tier 2 AI investigation",
    )
    rca_ai_provider: Optional[str] = Field(
        default=None,
        description="Optional override provider for Tier 2 AI investigation",
    )
    rca_ai_model: Optional[str] = Field(
        default=None,
        description="Optional override model identifier for Tier 2 AI investigation",
    )

    # AI Agent Architecture Configuration (Phase 5)
    agent_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Default timeout in seconds for agent task execution",
    )
    agent_max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum reasoning/action iterations permitted per agent task",
    )
    agent_read_only_mode: bool = Field(
        default=True,
        description="Enforce strict read-only safety boundary across all agents",
    )
    agent_max_input_chars: int = Field(
        default=16000,
        ge=500,
        le=100000,
        description="Maximum character limit for untrusted external agent context inputs",
    )

    # RAG Document Ingestion Configuration (Phase 5 Commit 2)
    rag_max_file_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
        description="Maximum allowed file size in bytes for RAG document ingestion (default: 10MB)",
    )
    rag_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Default character size per document chunk",
    )
    rag_chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=1000,
        description="Default character overlap between consecutive chunks",
    )
    rag_allowed_formats: List[str] = Field(
        default=[".md", ".txt", ".pdf", ".json", ".yaml", ".yml"],
        description="Allowed file extensions for RAG document ingestion",
    )

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"Ollama host must begin with http:// or https://: {v}")
        return v.rstrip("/")


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached global settings instance."""
    return Settings()
