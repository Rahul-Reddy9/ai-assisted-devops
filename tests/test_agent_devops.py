# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying DevOpsDiagnosticAgent execution, LLMRouter integration, and fallback."""

from unittest.mock import patch

from devops_ai.agents.devops_agent import DevOpsDiagnosticAgent
from devops_ai.agents.models import (
    AgentCapability,
    AgentRole,
    AgentStatus,
    AgentTask,
)
from devops_ai.core.exceptions import LLMProviderError
from devops_ai.llm.router import GenerationResult


def test_devops_agent_initialization():
    """Verify default diagnostic agent attributes and capabilities."""
    agent = DevOpsDiagnosticAgent()
    assert agent.name == "devops-diagnostician"
    assert agent.role == AgentRole.DIAGNOSTICIAN
    assert agent.status == AgentStatus.IDLE
    assert AgentCapability.INCIDENT_DIAGNOSIS in agent.capabilities
    assert AgentCapability.REMEDIATION_ADVISORY in agent.capabilities
    assert agent.limits.read_only is True


def test_devops_agent_llm_integration_mocked():
    """Verify LLMRouter is invoked with proper system prompt, temperature, and input framing."""
    agent = DevOpsDiagnosticAgent()

    plan_response = GenerationResult(
        content="1. Inspect application logs for unhandled exceptions\n2. Correlate memory utilization metrics",
        provider="mock-ollama",
        model="llama3.2:1b",
    )
    synthesis_response = GenerationResult(
        content="Memory saturation observed leading to pod termination. Recommend reviewing JVM heap allocation.",
        provider="mock-ollama",
        model="llama3.2:1b",
    )

    with patch(
        "devops_ai.agents.base.LLMRouter.generate_with_fallback",
        side_effect=[plan_response, synthesis_response],
    ) as mock_gen:
        task = AgentTask(
            description="Investigate sudden container restart on billing-service",
            context={
                "logs": ["OutOfMemoryError: Java heap space", "Process terminating"],
                "exit_code": 137,
                "metrics": {"memory_percent": 98.2},
            },
        )

        result = agent.execute(task)

        assert result.status == AgentStatus.COMPLETED
        assert result.agent_name == "devops-diagnostician"
        assert len(result.findings) >= 1
        assert "Memory saturation" in result.summary
        assert result.metadata.get("read_only") is True
        assert result.metadata.get("advisory_only") is True

        # Verify LLMRouter was called twice (plan + synthesis)
        assert mock_gen.call_count == 2
        # Verify first call args
        first_call = mock_gen.call_args_list[0]
        assert "STRICT READ-ONLY" in first_call.kwargs["system_prompt"]
        assert first_call.kwargs["temperature"] == 0.1


def test_devops_agent_graceful_fallback_without_llm():
    """Verify agent operates deterministically when LLM provider is unavailable."""
    agent = DevOpsDiagnosticAgent()

    with patch(
        "devops_ai.agents.base.LLMRouter.generate_with_fallback",
        side_effect=LLMProviderError("Connection refused to Ollama endpoint", provider="ollama"),
    ):
        task = AgentTask(
            description="Diagnose OOMKill alert",
            context={
                "logs": ["FATAL: OutOfMemoryError encountered"],
                "exit_code": 137,
                "metrics": {"mem_util": 0.99},
            },
        )

        result = agent.execute(task)

        assert result.status == AgentStatus.COMPLETED
        assert "Diagnostic review completed" in result.summary
        assert len(result.findings) >= 2
        # Check findings identified exit code 137
        categories = [f["category"] for f in result.findings]
        assert "CONTAINER" in categories
        assert "LOGS" in categories


def test_devops_agent_scrubs_secrets_in_context():
    """Verify secrets in task context are redacted from findings and observations."""
    agent = DevOpsDiagnosticAgent()

    with patch(
        "devops_ai.agents.base.LLMRouter.generate_with_fallback",
        side_effect=LLMProviderError("Offline"),
    ):
        # Dynamically constructed synthetic key to avoid static secret scanner false positives
        fake_api_key = "".join(["AIzaSy", "FakeSecretKey", "1234567890123456789012"])
        task = AgentTask(
            description="Check logs with sensitive tokens",
            context={
                "logs": [
                    f"Failed auth with key {fake_api_key}",
                    "Database password = 'super_secret_db_pass!'",
                ],
            },
        )

        result = agent.execute(task)

        assert result.status == AgentStatus.COMPLETED
        result_json = result.model_dump_json()
        assert "AIzaSy" not in result_json
        assert "[REDACTED_API_KEY]" in result_json
        assert "super_secret_db_pass" not in result_json
        assert "[REDACTED_SECRET]" in result_json
