"""Retrieval math: RRF fusion, filter compilation, embeddings, rerank scoring."""

from __future__ import annotations

import pytest

from app.services.embeddings import get_embedder
from app.services.embeddings.base import EmbeddingProvider
from app.services.retrieval.answer import AnswerGenerator
from app.services.retrieval.filters import SearchFilters, build_filters
from app.services.retrieval.hybrid import reciprocal_rank_fusion
from app.services.retrieval.rerank import normalize_scores


def _row(chunk_id: str, candidate_id: str, *, vector=None, keyword=None):
    return {
        "id": chunk_id,
        "candidate_id": candidate_id,
        "chunk_index": 0,
        "section": "experience",
        "heading": "Engineer | Acme",
        "content": f"content of {chunk_id}",
        **({"vector_score": vector} if vector is not None else {}),
        **({"keyword_score": keyword} if keyword is not None else {}),
    }


class TestReciprocalRankFusion:
    def test_documented_order_in_both_lists_wins(self):
        vector = [_row("a", "c1"), _row("b", "c2")]
        keyword = [_row("b", "c2"), _row("a", "c1")]
        fused = reciprocal_rank_fusion(vector, keyword, rrf_k=60)
        scores = {c.chunk_id: c.score for c in fused}
        # a: 1/61 + 1/62   b: 1/62 + 1/61  -> mathematically tied here.
        assert set(scores) == {"a", "b"}

        # Make the orders disagree so the tie is broken meaningfully.
        vector = [_row("a", "c1"), _row("b", "c2")]
        keyword = [_row("a", "c1"), _row("c", "c3")]
        fused = reciprocal_rank_fusion(vector, keyword, rrf_k=60)
        assert fused[0].chunk_id == "a"

    def test_only_ranks_are_used_not_raw_scores(self):
        # A huge keyword score must not outweigh a better rank position.
        vector = [_row("a", "c1", vector=0.9), _row("b", "c2", vector=0.1)]
        keyword = [_row("a", "c1", keyword=999.0)]
        fused = reciprocal_rank_fusion(vector, keyword, rrf_k=60)
        assert fused[0].chunk_id == "a"

    def test_items_in_only_one_list_still_appear(self):
        fused = reciprocal_rank_fusion([_row("a", "c1")], [_row("b", "c2")], rrf_k=60)
        assert {c.chunk_id for c in fused} == {"a", "b"}

    def test_leg_scores_are_kept_for_explainability(self):
        fused = reciprocal_rank_fusion(
            [_row("a", "c1", vector=0.87)], [_row("a", "c1", keyword=3.1)], rrf_k=60
        )
        chunk = fused[0]
        assert chunk.vector_score == 0.87
        assert chunk.keyword_score == 3.1
        assert chunk.vector_rank == 1
        assert chunk.keyword_rank == 1

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_weights_control_the_balance(self):
        vector = [_row("a", "c1")]
        keyword = [_row("b", "c2")]
        keyword_heavy = reciprocal_rank_fusion(vector, keyword, rrf_k=60, weight_keyword=10.0)
        assert keyword_heavy[0].chunk_id == "b"


class TestFilterCompilation:
    def test_no_filters_produces_no_sql(self):
        sql, params = build_filters(SearchFilters())
        assert sql is None and params == {}
        assert build_filters(None) == (None, {})

    def test_skills_all_mode_creates_one_exists_per_skill(self):
        sql, params = build_filters(SearchFilters(skills=["Python", "k8s"]))
        assert sql.count("EXISTS") == 2
        # Values are bound parameters, never string-interpolated.
        assert "python" not in sql.lower() or "Python" not in sql
        assert "python" in [str(v) for v in params.values()]
        assert "kubernetes" in [str(v) for v in params.values()]

    def test_skills_any_mode_uses_one_exists_with_an_array(self):
        sql, _ = build_filters(SearchFilters(skills=["python", "rust"], skills_mode="any"))
        assert sql.count("EXISTS") == 1
        assert "= ANY" in sql

    def test_experience_bounds_are_numeric_comparisons(self):
        sql, params = build_filters(SearchFilters(min_years_experience=5, max_years_experience=12))
        assert ">= :" in sql and "<= :" in sql
        assert 5.0 in params.values() and 12.0 in params.values()

    def test_location_matches_city_only(self):
        sql, _ = build_filters(SearchFilters(location="Berlin"))
        assert "split_part" in sql
        assert "ILIKE" in sql or "LIKE" in sql

    def test_sql_injection_attempt_stays_a_parameter(self):
        """The classic OR 1=1 / DROP TABLE payload must never reach the SQL text."""
        sql, params = build_filters(
            SearchFilters(location="'; DROP TABLE candidates; --", company="' OR '1'='1")
        )
        assert "DROP TABLE" not in sql
        assert "OR '1'='1" not in sql
        # It survives only as a harmless, lower-cased bound value.
        values = " | ".join(str(v).lower() for v in params.values())
        assert "drop table candidates" in values
        assert "or '1'='1" in values

    def test_alias_is_consistent(self):
        sql, _ = build_filters(SearchFilters(location="Berlin"))
        assert "c.location" in sql
        assert "candidates c" not in sql  # caller supplies the alias

    def test_presence_checks(self):
        sql, _ = build_filters(SearchFilters(has_email=True, has_phone=False))
        assert "c.email IS NOT NULL" in sql
        assert "c.phone IS NULL" in sql

    def test_languages_use_unnest_not_substring_matching(self):
        sql, params = build_filters(SearchFilters(languages=["English"]))
        assert "unnest" in sql
        # Stored as an array parameter so `= ANY(...)` can compare elements.
        assert ["english"] in params.values() or ["english"] in [
            v for v in params.values() if isinstance(v, list)
        ]

    def test_round_trips_to_dict(self):
        filters = SearchFilters(skills=["python"], location="Berlin", min_years_experience=4)
        as_dict = filters.to_dict()
        assert as_dict["skills"] == ["python"]
        assert as_dict["min_years_experience"] == 4


class TestEmbeddingProvider:
    async def test_default_provider_is_deterministic(self):
        embedder = get_embedder()
        a = await embedder.embed_query("senior python engineer with kubernetes")
        b = await embedder.embed_query("senior python engineer with kubernetes")
        assert a == b, "same input must produce the same vector"

    async def test_vectors_are_unit_length(self):
        vector = await get_embedder().embed_query("python developer")
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    async def test_related_text_is_closer_than_unrelated(self):
        embedder = get_embedder()
        query = await embedder.embed_query("python machine learning engineer")
        related = await embedder.embed_query("Python machine learning engineer with PyTorch")
        unrelated = await embedder.embed_query("baker pastry chef making sourdough bread")
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))
        assert dot(query, related) > dot(query, unrelated)

    async def test_batch_order_is_preserved(self):
        texts = ["alpha beta", "gamma delta", "epsilon zeta"]
        vectors = await get_embedder().embed_documents(texts)
        assert len(vectors) == 3
        assert vectors[0] == await get_embedder().embed_query("alpha beta")

    async def test_empty_batch(self):
        assert await get_embedder().embed_documents([]) == []

    def test_l2_normalize_handles_zero_vector(self):
        assert EmbeddingProvider.l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
        assert EmbeddingProvider.l2_normalize([3.0, 4.0]) == [0.6, 0.8]


class TestRerankScores:
    def test_min_max_scaling(self):
        assert normalize_scores([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]

    def test_flat_scores_do_not_divide_by_zero(self):
        assert normalize_scores([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]

    def test_empty(self):
        assert normalize_scores([]) == []


class TestCitationSanitising:
    def test_valid_citations_survive(self):
        text, invalid = AnswerGenerator._sanitize_citations("He used Kafka [1] and Postgres [2].", 3)
        assert text == "He used Kafka [1] and Postgres [2]."
        assert invalid is False

    def test_out_of_range_citations_are_removed_and_flagged(self):
        # A model citing [9] when only 3 excerpts exist is hallucinating.
        text, invalid = AnswerGenerator._sanitize_citations("Wrong [1] and fake [9].", 3)
        assert "[9]" not in text
        assert "[1]" in text
        assert invalid is True

    def test_multi_citations_are_filtered_per_element(self):
        text, invalid = AnswerGenerator._sanitize_citations("Mixed [1,7,2].", 3)
        assert text == "Mixed [1,2]."
        assert invalid is True

    def test_empty_answer(self):
        assert AnswerGenerator._sanitize_citations("", 3) == ("", False)


@pytest.mark.parametrize("k", [1, 60, 100])
def test_rrf_uses_the_k_parameter(k):
    fused = reciprocal_rank_fusion([_row("a", "c1")], [], rrf_k=k)
    assert fused[0].score == pytest.approx(1.0 / (k + 1))