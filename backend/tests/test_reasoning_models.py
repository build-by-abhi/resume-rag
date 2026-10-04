"""
Reasoning-model handling: `enable_thinking` and `<think>` stripping.

`Qwen/Qwen3-*` and `DeepSeek-R1-*` are hybrid reasoning models. Left on, they
emit a `<think>` block that breaks JSON extraction and gets rendered to the user.
These tests pin down the two defences:

  1. `llm_extra_body_merged` turns thinking OFF automatically for Qwen3
  2. `strip_reasoning` cleans up if the flag is ignored anyway
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.llm.client import parse_json_blob, strip_reasoning


def _settings(model: str = "Qwen/Qwen3-8B", **overrides) -> Settings:
    base = {"llm_model": model}
    base.update(overrides)
    return Settings(**base)


class TestExtraBody:
    def test_thinking_is_disabled_for_qwen3(self):
        """The whole point: Qwen3 must not emit a <think> block by default."""
        assert _settings("Qwen/Qwen3-8B").llm_extra_body_merged == {
            "enable_thinking": False
        }

    @pytest.mark.parametrize(
        "model", ["Qwen/Qwen3-14B", "Qwen/Qwen3-30B-A3B", "qwen/qwen3-8b"]
    )
    def test_applies_to_the_whole_qwen3_family(self, model):
        assert _settings(model).llm_extra_body_merged["enable_thinking"] is False

    @pytest.mark.parametrize("model", ["Qwen/Qwen2.5-7B-Instruct", "gpt-4o-mini"])
    def test_non_reasoning_models_get_nothing(self, model):
        # Qwen2.5 has no thinking mode; sending the flag could be rejected.
        assert _settings(model).llm_extra_body_merged == {}

    def test_explicit_override_wins(self):
        """A user who wants reasoning ON for hard problems can set it."""
        settings = _settings(
            "Qwen/Qwen3-8B", llm_extra_body='{"enable_thinking": true}'
        )
        assert settings.llm_extra_body_merged == {"enable_thinking": True}

    def test_extra_body_is_merged_not_replaced(self):
        settings = _settings(
            "Qwen/Qwen3-8B", llm_extra_body='{"top_k": 20, "enable_thinking": true}'
        )
        merged = settings.llm_extra_body_merged
        assert merged == {"top_k": 20, "enable_thinking": True}

    def test_invalid_json_is_ignored_not_fatal(self):
        """A typo in .env must not take the whole app down."""
        settings = _settings("Qwen/Qwen3-8B", llm_extra_body="{not json")
        assert settings.llm_extra_body_merged == {"enable_thinking": False}

    def test_json_array_is_ignored(self):
        settings = _settings("Qwen/Qwen3-8B", llm_extra_body="[1,2,3]")
        assert settings.llm_extra_body_merged == {"enable_thinking": False}

    def test_none_yields_dict_for_qwen3(self):
        assert isinstance(_settings("Qwen/Qwen3-8B").llm_extra_body_merged, dict)


class TestStripReasoning:
    def test_removes_a_complete_think_block(self):
        text = '<think>The candidate has 8 years.</think>{"skills": ["python"]}'
        assert strip_reasoning(text) == '{"skills": ["python"]}'

    def test_removes_thinking_and_think_tags(self):
        for tag in ("thinking", "reasoning", "THINK"):
            raw = f"<{tag}>reasoning here</{tag}>the answer"
            assert strip_reasoning(raw) == "the answer"

    def test_handles_multiline_reasoning(self):
        raw = "<think>\nline one\nline two\n</think>\n\nFinal answer"
        assert strip_reasoning(raw) == "Final answer"

    def test_removes_an_unterminated_block(self):
        """
        max_tokens can cut the trace off mid-sentence. A dangling opening tag
        would otherwise swallow the JSON that follows.
        """
        raw = '{"skills": ["python"]}<think>now let me reconsider all of this'
        assert strip_reasoning(raw) == '{"skills": ["python"]}'

    def test_leaves_ordinary_text_untouched(self):
        for raw in [
            "The answer is 42.",
            '{"skills": ["python"]}',
            "Use <b>bold</b> and <br/> tags",
            "",
        ]:
            assert strip_reasoning(raw) == raw.strip()

    def test_stripped_output_is_parseable(self):
        raw = '<think>long chain of thought</think>{"full_name": "Jane Doe"}'
        assert parse_json_blob(strip_reasoning(raw)) == {"full_name": "Jane Doe"}

    def test_json_survives_even_without_stripping(self):
        """
        Belt and braces: `parse_json_blob` digs the object out even if a caller
        forgets to strip. Extraction must not fail because of reasoning text.
        """
        raw = '<think>reasoning</think>{"full_name": "Jane Doe"}'
        assert parse_json_blob(raw) == {"full_name": "Jane Doe"}

    def test_none_input_is_safe(self):
        assert strip_reasoning(None) == ""