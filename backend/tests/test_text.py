"""Text cleaning + normalisation."""

from __future__ import annotations

from app.utils.text import (
    approx_token_count,
    clean_text,
    extract_emails,
    extract_urls,
    truncate,
)


class TestCleanText:
    def test_repairs_words_broken_across_lines(self):
        # The classic PDF failure: a soft hyphen plus a newline mid-word.
        assert "experience" in clean_text("3 years of experi-\nence in backend")

    def test_normalises_newlines_and_collapses_blank_runs(self):
        raw = "Line one\r\nLine two\r\rLine three\n\n\n\nLine four"
        assert "\r" not in clean_text(raw)
        assert "\n\n\n" not in clean_text(raw)

    def test_strips_page_numbers_and_bullets(self):
        raw = "Skills\n\n- Python\n- Docker\n\nPage 2 of 3\n\n1"
        cleaned = clean_text(raw)
        assert "Page 2 of 3" not in cleaned
        assert cleaned.startswith("Skills")
        # Bullets removed but the words survive.
        assert "Python" in cleaned and "Docker" in cleaned
        assert not cleaned.startswith("-")

    def test_normalises_ligatures_and_zero_width(self):
        # "efﬁcient" uses the "ffi" ligature -> "efficient".
        assert clean_text("e\uFB03cient") == "efficient"
        # "ﬁ" alone becomes "fi" (e.g. "proﬁcient" -> "proficient").
        assert clean_text("pro\uFB01cient") == "proficient"
        assert clean_text("a\u200bb").strip() == "ab"

    def test_folds_accents_for_forgiving_matching(self):
        # "José" -> "Jose" means a search for "Jose" still matches.
        assert "Jose Gonzalez" in clean_text("José González")

    def test_empty_input(self):
        assert clean_text("") == ""


class TestExtractors:
    def test_emails(self):
        found = extract_emails("Contact: jane.doe+work@example.co.uk today")
        assert found == ["jane.doe+work@example.co.uk"]

    def test_urls_are_trimmed_of_trailing_punctuation(self):
        urls = extract_urls("See https://example.com/path. Then (linkedin.com/in/jane)")
        assert urls == ["https://example.com/path", "linkedin.com/in/jane"]


class TestHelpers:
    def test_token_estimate_is_roughly_four_chars(self):
        assert approx_token_count("") == 0
        assert approx_token_count("a" * 400) == 100

    def test_truncate_adds_suffix_only_when_needed(self):
        assert truncate("short", 100) == "short"
        long = "x" * 300
        assert truncate(long, 50).endswith("...")
        assert len(truncate(long, 50)) <= 50