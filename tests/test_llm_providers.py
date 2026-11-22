# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for pluggable LLM provider services and router."""

from unittest.mock import MagicMock, patch
import pytest
from devops_ai.core.config import Settings
from devops_ai.core.exceptions import ConfigurationError, LLMProviderError
from devops_ai.llm.base import BaseLLMProvider
from devops_ai.llm.gemini_provider import GeminiProvider
from devops_ai.llm.ollama_provider import OllamaProvider
from devops_ai.llm.router import GenerationResult, LLMRouter, get_llm_provider


def test_base_provider_contract():
    class DummyProvider(BaseLLMProvider):
        def generate(self, prompt, system_prompt=None, temperature=0.2, **kwargs):
            return "dummy response"

        def check_health(self, verify_remote=False):
            return True

        @property
        def provider_name(self):
            return "dummy"

        @property
        def model_name(self):
            return "dummy-model"

        @property
        def host(self):
            return None

        def is_configured(self):
            return True

    provider = DummyProvider()
    assert provider.provider_name == "dummy"
    assert provider.model_name == "dummy-model"
    assert provider.host is None
    assert provider.is_configured() is True
    assert provider.generate("hello") == "dummy response"
    assert provider.check_health() is True


def test_ollama_provider_initialization():
    provider = OllamaProvider(model="test-model", host="http://my-host:11434", timeout=45.0)
    assert provider.provider_name == "ollama"
    assert provider.model_name == "test-model"
    assert provider.host == "http://my-host:11434"
    assert provider.is_configured() is True


def test_ollama_health_check_local_does_not_hit_network():
    provider = OllamaProvider(model="test-model", host="http://localhost:11434")
    with patch("urllib.request.urlopen") as mock_urlopen:
        result = provider.check_health(verify_remote=False)
        assert result is True
        mock_urlopen.assert_not_called()


def test_ollama_health_check_remote_probes_network():
    provider = OllamaProvider(model="test-model", host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        result = provider.check_health(verify_remote=True)
        assert result is True
        mock_urlopen.assert_called_once()


def test_ollama_health_check_remote_failure():
    provider = OllamaProvider(model="test-model", host="http://localhost:11434")
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        result = provider.check_health(verify_remote=True)
        assert result is False


def test_ollama_get_installed_models():
    provider = OllamaProvider(model="llama3.2:1b", host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"models": [{"name": "llama3.2:1b"}, {"name": "mistral:latest"}]}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = provider.get_installed_models()
        assert models == ["llama3.2:1b", "mistral:latest"]


def test_ollama_provider_mocked_chat(monkeypatch):
    mock_ollama = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.chat.return_value = {"message": {"content": "FROM python:3.12-slim"}}
    mock_ollama.Client.return_value = mock_client_instance
    monkeypatch.setattr("devops_ai.llm.ollama_provider.urllib.request", MagicMock())

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        provider = OllamaProvider(model="llama3.2:1b", host="http://localhost:11434")
        result = provider.generate("Generate Dockerfile", system_prompt="System instructions")
        assert result == "FROM python:3.12-slim"
        mock_client_instance.chat.assert_called_once()


def test_ollama_provider_sanitizes_exceptions(monkeypatch):
    mock_ollama = MagicMock()
    mock_client_instance = MagicMock()
    # Simulate an error message containing an API key
    mock_client_instance.chat.side_effect = Exception("Failed connecting to AIzaSyFakeKey123456789012345678901234")
    mock_ollama.Client.return_value = mock_client_instance

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        provider = OllamaProvider()
        with pytest.raises(LLMProviderError) as exc_info:
            provider.generate("Test prompt")
        assert "AIzaSyFakeKey" not in str(exc_info.value)
        assert "[REDACTED_API_KEY]" in str(exc_info.value)


def test_gemini_provider_is_configured():
    provider = GeminiProvider()
    # Default settings has google_api_key = None
    assert provider.is_configured() is False

    with patch("devops_ai.llm.gemini_provider.get_settings", return_value=Settings(google_api_key="valid-key")):
        configured_provider = GeminiProvider()
        assert configured_provider.is_configured() is True


def test_gemini_missing_api_key_raises_configuration_error():
    provider = GeminiProvider()
    with pytest.raises(ConfigurationError) as exc_info:
        provider.generate("Hello")
    assert "GOOGLE_API_KEY is not configured" in str(exc_info.value)


def test_gemini_timeout_conversion_to_milliseconds(monkeypatch):
    mock_google = MagicMock()
    mock_genai = MagicMock()
    mock_types = MagicMock()
    mock_google.genai = mock_genai
    mock_genai.types = mock_types
    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Generated content"
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_genai.Client.return_value = mock_client_instance

    with patch("devops_ai.llm.gemini_provider.get_settings", return_value=Settings(google_api_key="valid-key", llm_timeout_seconds=60.0)):
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai, "google.genai.types": mock_types}):
            provider = GeminiProvider(timeout=60.0)
            result = provider.generate("Hello")
            assert result == "Generated content"
            # Verify timeout conversion 60.0s -> 60000ms
            mock_types.HttpOptions.assert_called_once_with(timeout=60000)


def test_gemini_check_health_local_vs_remote():
    with patch("devops_ai.llm.gemini_provider.get_settings", return_value=Settings(google_api_key="valid-key")):
        provider = GeminiProvider()
        # Local check: verify_remote=False -> returns True without calling client
        assert provider.check_health(verify_remote=False) is True

        # Remote check: verify_remote=True with mocked client
        mock_google = MagicMock()
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_google.genai = mock_genai
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai, "google.genai.types": mock_types}):
            assert provider.check_health(verify_remote=True) is True
            mock_client.models.get.assert_called_once_with(model=provider.model_name)


def test_router_get_provider():
    ollama_prov = LLMRouter.get_provider("ollama")
    assert isinstance(ollama_prov, OllamaProvider)

    gemini_prov = LLMRouter.get_provider("gemini")
    assert isinstance(gemini_prov, GeminiProvider)

    with pytest.raises(ConfigurationError):
        LLMRouter.get_provider("unsupported-provider")


def test_router_helper_function():
    prov = get_llm_provider("ollama")
    assert isinstance(prov, OllamaProvider)


def test_generate_with_fallback_returns_structured_metadata():
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.provider_name = "ollama"
    mock_provider.model_name = "llama3.2:1b"
    mock_provider.generate.return_value = "Generated text"

    with patch.object(LLMRouter, "get_provider", return_value=mock_provider):
        res = LLMRouter.generate_with_fallback("Test prompt", allow_fallback=False)
        assert isinstance(res, GenerationResult)
        assert res.content == "Generated text"
        assert res.provider == "ollama"
        assert res.model == "llama3.2:1b"
        assert res.fallback_used is False


def test_generate_with_fallback_triggers_alternate_on_primary_failure():
    primary_mock = MagicMock(spec=BaseLLMProvider)
    primary_mock.provider_name = "ollama"
    primary_mock.generate.side_effect = LLMProviderError("Ollama failed", provider="ollama")

    fallback_mock = MagicMock(spec=BaseLLMProvider)
    fallback_mock.provider_name = "gemini"
    fallback_mock.model_name = "gemini-2.5-flash"
    fallback_mock.generate.return_value = "Fallback text"

    def mock_get_provider(provider_name, **kwargs):
        if provider_name == "ollama":
            return primary_mock
        return fallback_mock

    with patch.object(LLMRouter, "get_provider", side_effect=mock_get_provider):
        res = LLMRouter.generate_with_fallback("Test prompt", provider_name="ollama", allow_fallback=True)
        assert res.content == "Fallback text"
        assert res.provider == "gemini"
        assert res.fallback_used is True


def test_generate_with_fallback_handles_unexpected_primary_exception():
    primary_mock = MagicMock(spec=BaseLLMProvider)
    primary_mock.provider_name = "ollama"
    primary_mock.generate.side_effect = RuntimeError("Unexpected connection reset")

    fallback_mock = MagicMock(spec=BaseLLMProvider)
    fallback_mock.provider_name = "gemini"
    fallback_mock.model_name = "gemini-2.5-flash"
    fallback_mock.generate.return_value = "Recovered text"

    def mock_get_provider(provider_name, **kwargs):
        if provider_name == "ollama":
            return primary_mock
        return fallback_mock

    with patch.object(LLMRouter, "get_provider", side_effect=mock_get_provider):
        res = LLMRouter.generate_with_fallback("Test prompt", provider_name="ollama", allow_fallback=True)
        assert res.content == "Recovered text"
        assert res.fallback_used is True


def test_generate_with_fallback_fails_both_providers():
    primary_mock = MagicMock(spec=BaseLLMProvider)
    primary_mock.provider_name = "ollama"
    primary_mock.generate.side_effect = LLMProviderError("Ollama down", provider="ollama")

    fallback_mock = MagicMock(spec=BaseLLMProvider)
    fallback_mock.provider_name = "gemini"
    fallback_mock.generate.side_effect = LLMProviderError("Gemini quota exceeded", provider="gemini")

    def mock_get_provider(provider_name, **kwargs):
        if provider_name == "ollama":
            return primary_mock
        return fallback_mock

    with patch.object(LLMRouter, "get_provider", side_effect=mock_get_provider):
        with pytest.raises(LLMProviderError) as exc_info:
            LLMRouter.generate_with_fallback("Test prompt", provider_name="ollama", allow_fallback=True)
        assert "Primary provider 'ollama' failed" in str(exc_info.value)
        assert "fallback provider 'gemini' also failed" in str(exc_info.value)


def test_fallback_error_sanitization_masks_api_keys():
    primary_mock = MagicMock(spec=BaseLLMProvider)
    primary_mock.provider_name = "gemini"
    primary_mock.generate.side_effect = Exception("API error with key AIzaSyFakeKey123456789012345678901234 in URL")

    with patch.object(LLMRouter, "get_provider", return_value=primary_mock):
        with pytest.raises(LLMProviderError) as exc_info:
            LLMRouter.generate_with_fallback("Test", provider_name="gemini", allow_fallback=False)
        assert "AIzaSyFakeKey" not in str(exc_info.value)
        assert "[REDACTED_API_KEY]" in str(exc_info.value)
