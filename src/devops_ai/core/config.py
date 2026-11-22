# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Type-safe configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal, Optional
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

    # Observability defaults (Phase 1 preserved)
    anomaly_contamination: float = Field(default=0.1, ge=0.01, le=0.5)
    burst_threshold: int = Field(default=3, ge=1)
    burst_window_seconds: int = Field(default=30, ge=5)

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
