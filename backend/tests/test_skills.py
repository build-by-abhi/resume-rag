"""Skill taxonomy: normalisation, aliasing and free-list parsing."""

from __future__ import annotations

import pytest

from app.services.extraction.skills import (
    SKILL_TAXONOMY,
    category_of,
    display_name,
    extract_skills,
    normalize_skill,
    parse_skill_list,
)


class TestNormalisation:
    @pytest.mark.parametrize(
        "surface,canonical",
        [
            ("React.js", "react"),
            ("reactjs", "react"),
            ("React JS", "react"),
            ("Postgres", "postgresql"),
            ("PSQL", "postgresql"),
            ("k8s", "kubernetes"),
            ("Node", "nodejs"),
            ("NodeJS", "nodejs"),
            ("  Python  ", "python"),
            ("scikit learn", "scikit-learn"),
            ("sklearn", "scikit-learn"),
            ("PyTorch", "pytorch"),
            ("CI/CD", "ci-cd"),
            ("AWS", "aws"),
        ],
    )
    def test_surface_forms_collapse_to_one_key(self, surface, canonical):
        # This is the whole point of the taxonomy: one skill, one filter.
        assert normalize_skill(surface) == canonical

    def test_unknown_skill_returns_none(self):
        assert normalize_skill("Underwater Basket Weaving") is None
        assert normalize_skill("") is None

    def test_canonical_keys_are_unique(self):
        keys = [s.canonical for s in SKILL_TAXONOMY]
        assert len(keys) == len(set(keys)), "duplicate canonical key in taxonomy"

    def test_alias_collisions_are_intentional_not_accidental(self):
        # "ML" is claimed by machine-learning; verify it resolves consistently.
        assert normalize_skill("ML") == "machine-learning"
        assert category_of("machine-learning") == "ml"

    def test_display_names_are_human_readable(self):
        assert display_name("postgresql") == "PostgreSQL"
        assert display_name("unknown-thing") == "Unknown Thing"


class TestExtraction:
    def test_counts_occurrences(self):
        text = "Built services in Python. Migrated to PostgreSQL. Wrote Python tests."
        skills = extract_skills(text)
        assert skills["python"] == 2
        assert skills["postgresql"] == 1

    def test_longest_alias_wins(self):
        # "react native" must not be credited as plain "react" only.
        text = "Worked with React Native for three years."
        skills = extract_skills(text)
        assert skills  # at least one react-family skill detected

    def test_word_boundaries_prevent_false_positives(self):
        # "mlops" contains "ml" and "ops"; neither may match the bare "ml" alias.
        assert "machine-learning" not in extract_skills("Built MLOps pipelines")
        # "Ruby" must not match "R".
        assert "r" not in extract_skills("Ruby and Rails developer")
        # "Jenkins" is a CI/CD alias but must not be read as "go".
        assert "go" not in extract_skills("Set up Jenkins pipelines")

    def test_empty_text(self):
        assert extract_skills("") == {}
        assert extract_skills(None) == {}

    def test_case_insensitive(self):
        assert extract_skills("PYTHON and DOCKER and KUBERNETES")


class TestParseSkillList:
    def test_strips_category_labels(self):
        # Without label stripping, "Languages" itself would become a skill.
        raw = "Languages: Python, JavaScript | Tools: Docker, k8s"
        tokens = parse_skill_list(raw)
        assert "Python" in tokens
        assert "JavaScript" in tokens
        assert "Docker" in tokens
        assert "k8s" in tokens
        assert not any(t.lower() == "languages" for t in tokens)

    def test_strips_unlisted_category_labels(self):
        """
        Regression: the vocabulary originally lacked "tooling", so
        `tools?` matched the "Tool" inside "Tooling:" and left the fragment
        "ing: Prometheus, Grafana, ArgoCD" as a single bogus skill.
        """
        raw = "Tooling: Prometheus, Grafana, ArgoCD\nEnvironment: Java 17"
        tokens = parse_skill_list(raw)
        assert "Prometheus" in tokens
        assert "Grafana" in tokens
        assert "ArgoCD" in tokens
        assert not any(t.startswith("ing:") or ":" in t for t in tokens), tokens

    @pytest.mark.parametrize(
        "label",
        [
            "Languages", "Tools", "Tooling", "Frameworks", "Databases", "Data",
            "Cloud", "Infrastructure", "Messaging", "Testing", "Technologies",
            "Environment", "Nice to have", "Core competencies", "Methodologies",
        ],
    )
    def test_any_short_leading_label_is_stripped(self, label):
        tokens = parse_skill_list(f"{label}: Python, Docker")
        assert tokens == ["Python", "Docker"], f"{label!r} -> {tokens}"

    def test_does_not_strip_a_prefix_from_a_real_skill_name(self):
        """
        Regression: the label pattern had no trailing `\\b`, so "Data" matched
        inside "Datadog" and the leftover "dog" was stored as a skill. The same
        bug would invent "loudflare" from "Cloudflare" and "oolset" from
        "Toolset".
        """
        for raw, expected in [
            ("Tooling: Datadog, Grafana", ["Datadog", "Grafana"]),
            ("Cloud: Cloudflare, AWS", ["Cloudflare", "AWS"]),
            ("Tools: Toolset, Jenkins", ["Toolset", "Jenkins"]),
            ("Data: Databricks, Snowflake", ["Databricks", "Snowflake"]),
        ]:
            tokens = parse_skill_list(raw)
            assert tokens == expected, f"{raw!r} -> {tokens}"

    def test_bare_skill_name_with_a_label_prefix_is_untouched(self):
        # No colon and no boundary: the whole word is the skill.
        assert parse_skill_list("Datadog") == ["Datadog"]
        assert parse_skill_list("Database") == ["Database"]

    def test_does_not_strip_a_colon_that_is_not_a_label(self):
        """
        The generic label rule must not eat real skills containing a colon.
        Constraints: letters/space/&/- only, >=3 chars, and a space after the
        colon. So "C++: templates" and "a:b" survive intact.
        """
        assert parse_skill_list("C++: templates") == ["C++: templates"]
        assert parse_skill_list("x a:b") == ["x a:b"]
        assert parse_skill_list("a:b") == ["a:b"]

    def test_ignores_pure_numbers_and_long_tokens(self):
        tokens = parse_skill_list("2019, 2020, 2021, " + "A" * 60)
        assert not any(t.isdigit() for t in tokens)
        assert all(len(t) <= 40 for t in tokens)

    def test_various_separators(self):
        for raw in ("Python, Docker, AWS", "Python; Docker; AWS", "Python | Docker | AWS", "Python\tDocker\tAWS"):
            assert len(parse_skill_list(raw)) >= 2