# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Real-provider smoke tests.

These tests make live calls to external providers (Ollama or Google Gemini)
and are marked with @pytest.mark.smoke. They are deselected by default in
normal offline test runs (via pyproject.toml addopts = "-m 'not smoke'").

When executed explicitly (pytest -m smoke), each test checks whether the provider
is reachable/configured. If not available, the test skips gracefully using
pytest.skip(), ensuring zero false-positive test failures.
"""

import pytest
from devops_ai.llm.gemini_provider import GeminiProvider
from devops_ai.llm.ollama_provider import OllamaProvider


@pytest.mark.smoke
def test_real_ollama_provider_smoke():
    """Live smoke test for Ollama provider; skips gracefully if Ollama daemon or model is unavailable."""
    provider = OllamaProvider()
    if not provider.check_health(verify_remote=True):
        pytest.skip(f"Ollama daemon is not reachable at {provider.host}; skipping live smoke test.")

    installed_models = provider.get_installed_models()
    target_model = provider.model_name
    if not any(target_model == m or m.startswith(f"{target_model}:") for m in installed_models):
        pytest.skip(
            f"Configured Ollama model '{target_model}' is not installed at {provider.host} "
            f"(installed: {installed_models}). Run 'ollama pull {target_model}' to install."
        )

    response = provider.generate("Say 'hello' in one word.", temperature=0.1)
    assert response and len(response.strip()) > 0


@pytest.mark.smoke
def test_real_gemini_provider_smoke():
    """Live smoke test for Gemini provider; skips gracefully if API key is unset or unreachable."""
    provider = GeminiProvider()
    if not provider.is_configured():
        pytest.skip("GOOGLE_API_KEY is not configured; skipping Gemini live smoke test.")
    if not provider.check_health(verify_remote=True):
        pytest.skip("Gemini live remote verification failed; skipping live smoke test.")

    response = provider.generate("Say 'hello' in one word.", temperature=0.1)
    assert response and len(response.strip()) > 0
