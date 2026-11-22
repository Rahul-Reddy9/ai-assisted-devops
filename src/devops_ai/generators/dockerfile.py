# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""AI-powered Dockerfile generator with heuristic validation."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import List, Optional
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.core.file_io import atomic_write_text
from devops_ai.core.logging import get_logger
from devops_ai.llm.base import BaseLLMProvider
from devops_ai.llm.router import GenerationResult, LLMRouter

logger = get_logger("generators.dockerfile")

DOCKERFILE_SYSTEM_PROMPT = """You are a Principal Cloud Infrastructure & Container Security Architect.
Your task is to generate complete, production-grade, secure Dockerfiles for the requested application.

Strict output rules:
1. Output ONLY the raw Dockerfile content.
2. Do NOT enclose output in markdown backticks (no ``` or ```dockerfile).
3. Do NOT include introductory remarks, commentary, or conversational markdown.

Security & Quality Best Practices:
- Always use specific, official base images (e.g. python:3.12-slim, not python:latest).
- Use multi-stage builds where appropriate to minimize image attack surface.
- Never run containers as root. Explicitly create and switch to a non-root user (e.g. USER app or USER 1000).
- Leverage layer caching: copy dependency manifests first, install dependencies, then copy application code.
- Explicitly declare the EXPOSE port.
- Clean package manager caches (e.g. rm -rf /var/lib/apt/lists/*).
"""


@dataclass
class DockerfileValidationResult:
    """Result of Dockerfile static heuristic validation."""
    is_valid: bool
    content: str
    warnings: List[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    fallback_used: bool = False


class DockerfileGenerator:
    """Generates validated Dockerfiles using LLMs."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self.provider = provider

    def generate(
        self,
        language: str,
        port: int = 8000,
        base_image: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        output_path: Optional[Path] = None,
        force: bool = False,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        allow_fallback: Optional[bool] = None,
    ) -> DockerfileValidationResult:
        if type(port) is not int or isinstance(port, bool) or port < 1 or port > 65535:
            raise DevOpsAIError(
                f"Invalid port: {port}. Port must be a strict integer between 1 and 65535 (booleans rejected).",
                code="INVALID_ARGUMENT",
            )

        prompt = f"Generate a secure, production-grade Dockerfile for a '{language}' application. Target application port: {port}."
        if base_image:
            prompt += f" Required base image: '{base_image}'."
        if custom_instructions:
            prompt += f" Additional specifications: {custom_instructions}."

        if self.provider:
            raw_output = self.provider.generate(
                prompt=prompt,
                system_prompt=DOCKERFILE_SYSTEM_PROMPT,
                temperature=0.1,
            )
            gen_result = GenerationResult(
                content=raw_output,
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                fallback_used=False,
            )
        else:
            gen_result = LLMRouter.generate_with_fallback(
                prompt=prompt,
                system_prompt=DOCKERFILE_SYSTEM_PROMPT,
                provider_name=provider_name,
                model=model,
                host=host,
                timeout=timeout,
                allow_fallback=allow_fallback,
                temperature=0.1,
            )

        cleaned_content = self._clean_markdown_fences(gen_result.content)
        validation_result = self._validate_dockerfile(cleaned_content, requested_port=port)
        validation_result.provider = gen_result.provider
        validation_result.model = gen_result.model
        validation_result.fallback_used = gen_result.fallback_used

        if output_path:
            atomic_write_text(output_path, cleaned_content, force=force)
            logger.info(f"Saved generated Dockerfile to {output_path}")

        return validation_result

    @staticmethod
    def _clean_markdown_fences(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:dockerfile|docker|sh|bash)?\r?\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\r?\n```$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _validate_dockerfile(content: str, requested_port: Optional[int] = None) -> DockerfileValidationResult:
        """Heuristic validation of Dockerfile syntax and security patterns.

        Note: This is a static regex heuristic check that catches common errors,
        not a full Dockerfile parser or container vulnerability scanner.
        """
        if not content or not content.strip():
            raise DevOpsAIError("Generated Dockerfile is empty.", code="GENERATOR_EMPTY_OUTPUT")

        # Strip line comments before parsing instructions to avoid counting commented-out lines
        lines_without_comments = [
            line for line in content.splitlines() if not re.match(r"^\s*#", line)
        ]
        uncommented_content = "\n".join(lines_without_comments)

        # 1. Fatal Syntax Check: Must contain valid FROM instruction
        if not re.search(r"^\s*FROM\s+(?:--platform=\S+\s+)?\S+", uncommented_content, re.MULTILINE | re.IGNORECASE):
            raise DevOpsAIError(
                "Generated Dockerfile is missing required 'FROM' instruction.",
                code="GENERATOR_VALIDATION_ERROR",
            )

        # 2. Fatal Security Pattern Check: Disallow destructive commands
        fatal_patterns = [
            (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(?:/\s*$|/\*|/[^a-zA-Z0-9])", "Dangerous recursive root deletion 'rm -rf /' detected."),
            (r"\bchmod\s+(?:-[a-zA-Z]*\s+)?777\s+/", "Overly permissive root permission change 'chmod 777 /' detected."),
        ]
        for pattern, msg in fatal_patterns:
            if re.search(pattern, uncommented_content, re.MULTILINE):
                raise DevOpsAIError(
                    f"Generated Dockerfile contains hazardous command: {msg}",
                    code="GENERATOR_SECURITY_VIOLATION",
                )

        # 3. Static Heuristic Warnings
        warnings: List[str] = []

        # Multi-stage aware USER validation: evaluate USER specifically in the final build stage
        stages = re.split(r"(?=^\s*FROM\s+)", uncommented_content, flags=re.MULTILINE | re.IGNORECASE)
        stages = [s for s in stages if re.match(r"^\s*FROM\s+", s, re.IGNORECASE)]
        final_stage = stages[-1] if stages else uncommented_content

        user_matches = re.findall(r"^\s*USER\s+(\S+)", final_stage, re.MULTILINE | re.IGNORECASE)
        if not user_matches:
            warnings.append("No 'USER' instruction found in final build stage; container will run as root.")
        else:
            last_user = user_matches[-1]
            if re.match(r"^(?:root|0+|0+:0+|root:root|0+:root|root:0+)$", last_user, re.IGNORECASE):
                warnings.append(f"Final build stage explicitly configured with root user 'USER {last_user}'; recommend running as non-root.")

        if re.search(r"^\s*FROM\s+(?:--platform=\S+\s+)?(?:[\w./-]+):latest\b", uncommented_content, re.MULTILINE | re.IGNORECASE):
            warnings.append("Base image uses unpinned ':latest' tag; pin to a specific version or digest.")

        if re.search(r"\bsudo\b", uncommented_content, re.IGNORECASE):
            warnings.append("Reference to 'sudo' detected; avoid sudo in container environments.")

        if re.search(r"\b(curl|wget)\b.*\|\s*(ba)?sh", uncommented_content, re.IGNORECASE):
            warnings.append("Piping curl/wget directly to shell detected; verify remote script integrity.")

        if requested_port and not re.search(rf"^\s*EXPOSE\s+.*?\b{requested_port}\b", uncommented_content, re.MULTILINE | re.IGNORECASE):
            warnings.append(f"Requested port {requested_port} does not appear in an EXPOSE instruction.")

        for w in warnings:
            logger.warning(f"Dockerfile validation warning: {w}")

        return DockerfileValidationResult(
            is_valid=True,
            content=content,
            warnings=warnings,
        )
