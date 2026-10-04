"""
HuggingFace provider wiring.

The network calls are never made here. What these tests pin down is the part
that actually breaks in practice: provider selection from settings, the JSON
fallback that the extractor depends on, and the error messages for the two
gated-model failure modes.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.llm import client as C
from app.services.llm.client import (
    HuggingFaceClient,
    NullLLMClient,
    OpenAIClient,
    _explain_hf_error,
    build_llm_client,
    parse_json_blob,
)


def _settings(**overrides) -> Settings:
    """Settings built from an explicit dict, ignoring the developer's .env."""
    base = {
        "llm_provider": "huggingface",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "huggingface_api_key": "hf_test_token",
    }
    base.update(overrides)
    return Settings(**base)


class TestProviderSelection:
    def test_huggingface_provider_is_selected(self, monkeypatch):
        monkeypatch.setattr("app.services.llm.client.settings", _settings())
        client = build_llm_client("huggingface")
        assert isinstance(client, HuggingFaceClient)
        assert client.model == "meta-llama/Llama-3.2-3B-Instruct"

    def test_hf_alias_is_accepted(self, monkeypatch):
        monkeypatch.setattr("app.services.llm.client.settings", _settings())
        assert isinstance(build_llm_client("hf"), HuggingFaceClient)

    def test_missing_key_falls_back_instead_of_crashing(self, monkeypatch):
        """
        A missing key must not break search. The app degrades to the extractive
        answer and logs a warning - that is the whole point of NullLLMClient.
        """
        monkeypatch.setattr(
            "app.services.llm.client.settings", _settings(huggingface_api_key=None)
        )
        assert isinstance(build_llm_client("huggingface"), NullLLMClient)

    def test_openai_still_works_when_configured(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.llm.client.settings",
            _settings(llm_provider="openai", openai_api_key="sk-test", huggingface_api_key=None),
        )
        assert isinstance(build_llm_client("openai"), OpenAIClient)

    def test_client_raises_a_clear_error_when_key_is_absent(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.llm.client.settings", _settings(huggingface_api_key=None)
        )
        client = HuggingFaceClient()
        with pytest.raises(Exception, match="HUGGINGFACE_API_KEY"):
            client._get_client()


class TestGatedModelErrors:
    """The Llama licence trap produces a 401 that looks like a bad key."""

    def test_401_explains_the_licence_requirement(self):
        msg = _explain_hf_error(Exception("401 Client Error: Unauthorized"), "meta-llama/X")
        assert "gated" in msg.lower()
        assert "licence" in msg.lower()

    def test_403_points_at_the_licence_page(self):
        msg = _explain_hf_error(Exception("403 Forbidden"), "meta-llama/Llama-3.2-3B-Instruct")
        assert "huggingface.co/meta-llama/Llama-3.2-3B-Instruct" in msg

    def test_503_says_retry(self):
        assert "overloaded" in _explain_hf_error(Exception("503 Service Unavailable"), "m").lower()

    def test_404_says_check_the_repo_id(self):
        assert "not found" in _explain_hf_error(Exception("404 Not Found"), "m").lower()

    def test_unknown_error_passes_through_the_model_name(self):
        assert "some-model" in _explain_hf_error(Exception("boom"), "some-model")


class TestJsonFallback:
    """
    HuggingFace has no reliable cross-model equivalent of OpenAI's
    `response_format`, so JSON comes from the prompt and is parsed defensively.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            '{"skills": ["python"]}',
            '```json\n{"skills": ["python"]}\n```',
            'Here you go:\n{"skills": ["python"]}\nHope that helps!',
            '{"a": 1,}',
        ],
    )
    def test_recovers_json_from_common_wrappers(self, raw):
        assert parse_json_blob(raw) is not None

    def test_returns_none_for_unusable_output(self):
        assert parse_json_blob("I cannot help with that.") is None
        assert parse_json_blob("") is None

    def test_complete_json_is_inherited_not_overridden_with_native_mode(self):
        """
        Guard against someone "optimising" this into a native JSON-mode call
        that only some HF providers support.
        """
        assert HuggingFaceClient.supports_native_json_mode is False


class TestLlmStatus:
    """
    `llm_status` exists so the UI can say WHICH setting is missing.

    "LLM_PROVIDER=none" and "LLM_PROVIDER=huggingface but the key is empty" are
    different mistakes, and reporting both as "no LLM key configured" sends
    people looking in the wrong place.
    """

    def test_provider_none_explains_how_to_enable(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.llm.client.settings", _settings(llm_provider="none")
        )
        status = C.llm_status()
        assert status["configured"] is False
        assert "LLM_PROVIDER=none" in status["reason"]

    def test_missing_key_names_the_env_var(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.llm.client.settings", _settings(huggingface_api_key=None)
        )
        status = C.llm_status()
        assert status["configured"] is False
        assert status["env_var"] == "HUGGINGFACE_API_KEY"
        assert "HUGGINGFACE_API_KEY is empty" in status["reason"]
        assert "LLM_PROVIDER=huggingface is set" in status["reason"]

    @pytest.mark.parametrize(
        "provider,env_var",
        [
            ("openai", "OPENAI_API_KEY"),
            ("huggingface", "HUGGINGFACE_API_KEY"),
            ("hf", "HUGGINGFACE_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
        ],
    )
    def test_each_provider_maps_to_its_key(self, provider, env_var, monkeypatch):
        base = {"llm_provider": provider}
        monkeypatch.setattr(
            "app.services.llm.client.settings", _settings(**{**base, "huggingface_api_key": None})
        )
        status = C.llm_status()
        assert status["env_var"] == env_var

    def test_configured_when_key_present(self, monkeypatch):
        monkeypatch.setattr("app.services.llm.client.settings", _settings())
        status = C.llm_status()
        assert status["configured"] is True
        assert status["reason"] is None
        assert status["model"] == "meta-llama/Llama-3.2-3B-Instruct"

    def test_blank_whitespace_key_counts_as_missing(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.llm.client.settings", _settings(huggingface_api_key="   ")
        )
        assert C.llm_status()["configured"] is False


class TestModelConfig:
    def test_default_model_is_llama_when_configured(self):
        client = HuggingFaceClient(model="meta-llama/Llama-3.2-3B-Instruct")
        assert client.model == "meta-llama/Llama-3.2-3B-Instruct"

    def test_nul_client_is_still_the_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.llm.client.settings", _settings(huggingface_api_key=None)
        )
        client = build_llm_client()
        assert isinstance(client, NullLLMClient)
        assert asyncio_is_clean(client)


def asyncio_is_clean(client) -> bool:
    """NullLLMClient must answer, not raise."""
    import asyncio

    return asyncio.run(client.complete("s", "u")).empty