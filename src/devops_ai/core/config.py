# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Type-safe configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, field_validator
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

    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_model: str = Field(default="llama3.2:1b", description="Default local LLM model")

    # Hosted LLM settings (neutral configurable placeholder)
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    gemini_model: Optional[str] = Field(default=None, description="Configurable Gemini model identifier")

    # Observability defaults
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
