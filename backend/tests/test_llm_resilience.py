"""
Empty / truncated LLM responses.

Found in production: `Qwen/Qwen3-8B` prefixes its answer with whitespace, and a
tight `max_tokens` made the provider return `finish_reason="length"` with
`content=""`. That is indistinguishable from a dead key by inspection, so the
client retries with a larger budget and the answer layer falls back rather than
rendering an empty panel.

All tests use a stubbed transport - no network.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.llm.client import HuggingFaceClient, LLMError
from app.services.retrieval.answer import AnswerGenerator
from app.services.retrieval.hybrid import (
    CandidateHit,
    RetrievedChunk,
    SearchOutcome,
)


def _settings(**overrides) -> Settings:
    base = {
        "llm_provider": "huggingface",
        "llm_model": "Qwen/Qwen3-8B",
        "huggingface_api_key": "hf_test",
    }
    base.update(overrides)
    return Settings(**base)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeUsage:
    prompt_tokens = 30
    completion_tokens = 60


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = _FakeUsage()


class _StubClient:
    """Returns a scripted sequence of responses and records the budgets used."""

    def __init__(self, responses: list[tuple[str, str]]) -> None:
        self._responses = responses
        self.budgets: list[int] = []
        self.calls = 0

    def chat_completion(self, **kwargs):
        self.budgets.append(kwargs["max_tokens"])
        content, finish = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return _FakeResponse(content, finish)


def _client_with(responses, monkeypatch) -> tuple[HuggingFaceClient, _StubClient]:
    stub = _StubClient(responses)
    monkeypatch.setattr("app.services.llm.client.settings", _settings())
    client = HuggingFaceClient()
    monkeypatch.setattr(client, "_get_client", lambda: stub)
    return client, stub


class TestTruncationRetry:
    async def test_retries_when_truncated_to_nothing(self, monkeypatch):
        """
        The real failure. First call burns the budget on whitespace and returns
        empty content; the retry must recover a real answer.
        """
        client, stub = _client_with(
            [("", "length"), ("\n\nParis.", "stop")], monkeypatch
        )
        response = await client.complete("sys", "usr", max_tokens=60)

        assert response.text == "Paris."
        assert stub.calls == 2
        assert stub.budgets == [60, 240]

    async def test_no_retry_when_content_is_present(self, monkeypatch):
        client, stub = _client_with([("Paris.", "stop")], monkeypatch)
        await client.complete("sys", "usr", max_tokens=60)
        assert stub.calls == 1, "must not waste a second call"

    async def test_no_retry_on_a_normal_length_truncation(self, monkeypatch):
        """`finish_reason=length` with real content is a usable partial answer."""
        client, stub = _client_with([("Paris is the", "length")], monkeypatch)
        response = await client.complete("sys", "usr", max_tokens=60)
        assert response.text == "Paris is the"
        assert stub.calls == 1

    async def test_retry_budget_is_capped(self, monkeypatch):
        client, stub = _client_with([("", "length"), ("ok", "stop")], monkeypatch)
        await client.complete("sys", "usr", max_tokens=4000)
        assert stub.budgets[1] == 4096, "must not exceed the retry ceiling"

    async def test_gives_up_after_one_retry(self, monkeypatch):
        """Still empty after a retry -> return empty rather than loop forever."""
        client, stub = _client_with([("", "length")], monkeypatch)
        response = await client.complete("sys", "usr", max_tokens=60)
        assert response.empty
        assert stub.calls == 2

    async def test_leading_whitespace_is_stripped(self, monkeypatch):
        """Qwen3 prefixes "\\n\\n"; it would otherwise show as blank lines."""
        client, _ = _client_with([("\n\n\n  Paris.", "stop")], monkeypatch)
        response = await client.complete("sys", "usr", max_tokens=300)
        assert response.text == "Paris."

    async def test_finish_reason_is_exposed(self, monkeypatch):
        client, _ = _client_with([("Paris.", "stop")], monkeypatch)
        response = await client.complete("sys", "usr")
        assert response.finish_reason == "stop"
        assert response.truncated is False


class TestEmptyGuardrails:
    async def test_transport_errors_still_raise(self, monkeypatch):
        monkeypatch.setattr("app.services.llm.client.settings", _settings())
        client = HuggingFaceClient()

        class Boom:
            def chat_completion(self, **kwargs):
                raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(client, "_get_client", lambda: Boom())
        with pytest.raises(LLMError, match="gated"):
            await client.complete("sys", "usr")


class TestAnswerFallback:
    """An empty model response must never render as an empty answer panel."""

    class _EmptyClient:
        name = "stub"
        model = "stub-model"

        async def complete(self, system, user, *, max_tokens=700, temperature=None):
            from app.services.llm.client import LLMResponse

            return LLMResponse(text="", model="stub-model", finish_reason="length")

        async def complete_json(self, system, user, *, max_tokens=1200, temperature=0.0):
            return None

    def _outcome(self):
        chunk = RetrievedChunk(
            chunk_id="c1",
            candidate_id="p1",
            chunk_index=0,
            section="experience",
            heading=None,
            content="Staff Engineer at Zalando. Ran Kafka pipelines.",
        )

        class _C:
            id = "p1"
            full_name = "Priya Raghunathan"
            current_title = "Staff Backend Engineer"
            current_company = "Zalando"
            location = "Berlin, Germany"
            total_years_experience = 8.0
            summary = "Senior backend engineer."
            skill_links: list = []

        return SearchOutcome(
            hits=[CandidateHit(candidate=_C(), score=0.03, best_chunk=chunk)],
            chunks=[chunk],
        )

    async def test_falls_back_to_extractive_instead_of_empty(self):
        answer = await AnswerGenerator(client=self._EmptyClient()).answer(
            "who has kafka experience?", self._outcome()
        )
        assert answer.used_llm is False
        assert answer.text.strip(), "must not produce an empty answer"
        assert "Priya Raghunathan" in answer.text
        assert answer.citations, "fallback must still be citable"

    async def test_non_empty_response_is_used(self):
        class _GoodClient(TestAnswerFallback._EmptyClient):
            async def complete(self, system, user, *, max_tokens=700, temperature=None):
                from app.services.llm.client import LLMResponse

                return LLMResponse(text="Priya ran Kafka [1].", model="stub-model")

        answer = await AnswerGenerator(client=_GoodClient()).answer(
            "who has kafka experience?", self._outcome()
        )
        assert answer.used_llm is True
        assert "Kafka" in answer.text


class TestResponseFlags:
    def test_empty_and_truncated_helpers(self):
        from app.services.llm.client import LLMResponse

        assert LLMResponse(text="   ", model="m").empty is True
        assert LLMResponse(text="x", model="m").empty is False
        assert LLMResponse(text="x", model="m", finish_reason="length").truncated is True
        assert LLMResponse(text="x", model="m", finish_reason="stop").truncated is False