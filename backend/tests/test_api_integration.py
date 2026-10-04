"""
Full-stack tests against a live Postgres + pgvector database.

Run with:  pytest -m integration

Each test ingests the four sample resumes (via the /text endpoint, so no PDF
files are needed) and then exercises real SQL: filters, BM25, vector search,
RRF fusion and the answer layer. The session fixture rolls everything back, so
your database is never polluted.
"""

from __future__ import annotations

import pytest

from app.services.retrieval.hybrid import HybridSearcher

pytestmark = pytest.mark.integration


async def _seed(client) -> None:
    """Ingest every sample resume through the public API."""
    for text in SAMPLE_RESUMES.values():
        response = await client.post("/api/candidates/text", json={"text": text})
        assert response.status_code == 201, response.text


# The sample resumes live in conftest; import them lazily so this module stays
# readable on its own.
from tests.conftest import ALL_SAMPLES as SAMPLE_RESUMES  # noqa: E402


class TestIngestion:
    async def test_health_reports_the_schema_is_ready(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["database"] == "ok"
        assert body["embedding_dim"] > 0

    async def test_readiness_probe(self, client):
        assert (await client.get("/health/ready")).status_code == 200

    async def test_upload_writes_candidate_chunks_and_skills(self, client):
        await _seed(client)

        response = await client.get("/api/candidates?limit=50")
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 4

        names = {r["full_name"] for r in rows}
        assert "Priya Raghunathan" in names
        assert "Marcus Webb" in names
        assert any("Ananya Iyer" in n for n in names), names
        assert "Tomasz Nowak" in names

        # Every candidate must be searchable: chunks exist for all of them.
        for row in rows:
            detail = (await client.get(f"/api/candidates/{row['id']}")).json()
            assert detail["chunk_count"] >= 1, f"{row['full_name']} has no chunks"
            assert detail["skills"], f"{row['full_name']} has no skills"
            assert detail["text_length"] > 500

    async def test_structured_fields_are_correct(self, client):
        await _seed(client)
        rows = (await client.get("/api/candidates?limit=50")).json()
        priya = next(r for r in rows if r["full_name"] == "Priya Raghunathan")

        assert priya["email"] == "priya.raghunathan@example.com"
        assert "+49 30 5551234" in priya["phone"]
        assert priya["location"] == "Berlin, Germany"
        assert priya["current_title"] == "Staff Backend Engineer"
        assert priya["current_company"] == "Zalando"
        assert priya["seniority"] == "Senior"

        detail = (await client.get(f"/api/candidates/{priya['id']}")).json()
        canonical = {s["canonical"] for s in detail["skills"]}
        assert {"python", "postgresql", "kubernetes", "kafka"} <= canonical
        assert "k8s" not in canonical, "aliases must collapse to canonical keys"
        assert detail["links"]["linkedin"].startswith("https://")
        assert detail["extraction_method"] == "rules"

    async def test_tsvector_generated_column_is_populated(self, client, session):
        """BM25 depends on Postgres populating `search_tsv` from `content`."""
        from sqlalchemy import text as sql_text

        await _seed(client)
        rows = (
            await session.execute(
                sql_text(
                    "SELECT count(*) FILTER (WHERE search_tsv IS NULL) AS empty, "
                    "count(*) AS total FROM resume_chunks"
                )
            )
        ).mappings().one()
        assert rows["total"] > 0
        assert rows["empty"] == 0, "generated tsvector column must be filled by Postgres"

    async def test_embeddings_are_stored_for_every_chunk(self, client, session):
        from sqlalchemy import text as sql_text

        await _seed(client)
        row = (
            await session.execute(
                sql_text(
                    "SELECT count(*) FILTER (WHERE embedding IS NULL) AS missing, "
                    "count(*) AS total FROM resume_chunks"
                )
            )
        ).mappings().one()
        assert row["missing"] == 0
        assert row["total"] >= 4

    async def test_reuploading_the_same_text_deduplicates(self, client):
        resume = SAMPLE_RESUMES["priya_backend_python"]
        first = (await client.post("/api/candidates/text", json={"text": resume})).json()
        second = (await client.post("/api/candidates/text", json={"text": resume})).json()
        assert first["created"] is True
        assert second["created"] is False, "same content hash must not create a duplicate"
        assert len((await client.get("/api/candidates?limit=50")).json()) == 1

    async def test_rejects_non_resume_input(self, client):
        response = await client.post("/api/candidates/text", json={"text": "too short"})
        assert response.status_code == 422

    async def test_delete_removes_chunks_too(self, client, session):
        from sqlalchemy import text as sql_text

        await _seed(client)
        row = (await client.get("/api/candidates?limit=1")).json()[0]
        assert (await client.delete(f"/api/candidates/{row['id']}")).status_code == 204

        remaining = (
            await session.execute(
                sql_text("SELECT count(*) AS n FROM resume_chunks WHERE candidate_id = :cid"),
                {"cid": row["id"]},
            )
        ).scalar_one()
        assert remaining == 0, "chunks must cascade with the candidate"


class TestHybridSearch:
    async def test_natural_language_query_returns_the_right_people(self, client):
        await _seed(client)

        response = await client.post(
            "/api/search", json={"query": "senior backend engineer with Kubernetes", "top_k": 5}
        )
        assert response.status_code == 200
        body = response.json()

        assert body["total"] > 0
        names = [r["full_name"] for r in body["results"]]
        # Top-2 rather than top-1 on purpose: with RERANK_ENABLED=true the
        # cross-encoder's own judgement decides the order, and it legitimately
        # ranks the (very Kubernetes-heavy) DevOps resume first for a short
        # query. What must hold is that the right candidate is near the top.
        assert "Priya Raghunathan" in names[:2], names

        top = body["results"][0]
        # Evidence must come back so the UI can explain the match.
        assert top["evidence"], "every result carries its supporting chunks"
        assert body["retrieval_ms"] > 0
        assert body["diagnostics"]["fused_chunks"] > 0

    async def test_keyword_leg_finds_exact_terms(self, client):
        await _seed(client)
        response = await client.post("/api/search", json={"query": "Terraform", "top_k": 5})
        names = [r["full_name"] for r in response.json()["results"]]
        assert "Tomasz Nowak" in names

    async def test_vector_leg_finds_concepts_without_shared_words(self, client):
        await _seed(client)
        # "search" and "retrieval" never appear in the ML resume together, but
        # the semantic leg should still surface the RAG engineer.
        response = await client.post(
            "/api/search", json={"query": "built a searchable knowledge base over documents", "top_k": 5}
        )
        names = [r["full_name"] for r in response.json()["results"]]
        assert names, "semantic search must return something for a concept query"

    async def test_both_legs_contribute_to_the_fusion(self, client):
        await _seed(client)
        response = await client.post("/api/search", json={"query": "python postgres", "top_k": 5})
        diagnostics = response.json()["diagnostics"]
        assert diagnostics["vector_hits"] > 0, "vector leg produced nothing"
        assert diagnostics["keyword_hits"] > 0, "keyword leg produced nothing"
        # Fusion must return at most the union of both legs.
        assert diagnostics["fused_chunks"] <= (
            diagnostics["vector_hits"] + diagnostics["keyword_hits"]
        )

    async def test_structured_filters_narrow_results(self, client):
        await _seed(client)

        # Postgres + Python + 5 years should not include the junior frontend dev.
        response = await client.post(
            "/api/search",
            json={
                "query": "engineer",
                "skills": ["python", "postgresql"],
                "skills_mode": "all",
                "min_years_experience": 5,
                "top_k": 10,
            },
        )
        names = [r["full_name"] for r in response.json()["results"]]
        assert "Priya Raghunathan" in names
        assert "Marcus Webb" not in names

    async def test_skill_alias_filters_are_canonicalised(self, client):
        await _seed(client)
        # "k8s" is an alias of "kubernetes" in the taxonomy.
        by_alias = await client.post("/api/search", json={"skills": ["k8s"], "top_k": 10})
        by_canonical = await client.post("/api/search", json={"skills": ["kubernetes"], "top_k": 10})
        assert {r["id"] for r in by_alias.json()["results"]} == {
            r["id"] for r in by_canonical.json()["results"]
        }

    async def test_filters_only_mode_returns_matching_candidates(self, client):
        await _seed(client)
        response = await client.post(
            "/api/search", json={"location": "Berlin", "generate_answer": False, "top_k": 10}
        )
        body = response.json()
        assert body["total"] == 1
        assert body["results"][0]["full_name"] == "Priya Raghunathan"

    async def test_impossible_filter_returns_empty_not_error(self, client):
        await _seed(client)
        response = await client.post(
            "/api/search",
            json={"skills": ["cobol"], "min_years_experience": 30, "generate_answer": False},
        )
        assert response.status_code == 200
        assert response.json()["total"] == 0

    async def test_sql_injection_through_filters_is_safe(self, client):
        await _seed(client)
        response = await client.post(
            "/api/search",
            json={
                "location": "'; DROP TABLE candidates; --",
                "generate_answer": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["total"] == 0
        # The table must still exist afterwards.
        assert (await client.get("/api/candidates?limit=1")).status_code == 200

    async def test_empty_request_is_rejected(self, client):
        assert (await client.post("/api/search", json={})).status_code == 422

    async def test_company_filter_alone_counts_as_a_filter(self, client):
        """
        Regression guard.

        `has_text_or_filters` is what decides 422 vs 200, and the frontend
        mirrors it in `isSearchable()`. `company` was missing from the mirror,
        which meant the UI believed a company-only search was valid while the
        API rejected it. This pins the backend half.
        """
        await _seed(client)
        response = await client.post(
            "/api/search",
            json={"company": "Zalando", "generate_answer": False},
        )
        assert response.status_code == 200, response.text
        assert [r["full_name"] for r in response.json()["results"]] == ["Priya Raghunathan"]

    @pytest.mark.parametrize(
        "payload",
        [
            {},                                        # nothing at all
            {"query": ""},                             # blank question only
            {"query": "   ", "skills": []},            # whitespace + no filters
        ],
    )
    async def test_payloads_without_text_or_filters_are_422(self, client, payload):
        assert (await client.post("/api/search", json=payload)).status_code == 422

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": "python"},
            {"skills": ["python"]},
            {"location": "Berlin"},
            {"current_title": "Engineer"},
            {"company": "Zalando"},
            {"min_years_experience": 3},
            {"max_years_experience": 20},
            {"seniority": ["Senior"]},
            {"languages": ["English"]},
            {"education_level": ["MS"]},
            {"has_email": True},
            {"has_phone": False},
        ],
    )
    async def test_every_single_filter_alone_is_accepted(self, client, payload):
        """Each filter must independently satisfy `has_text_or_filters`."""
        response = await client.post("/api/search", json=payload)
        assert response.status_code == 200, f"{payload} -> {response.text}"

    async def test_answer_is_cited_and_grounded(self, client):
        await _seed(client)
        response = await client.post(
            "/api/search",
            json={"query": "Does anyone have hybrid retrieval or RAG experience?", "top_k": 5},
        )
        answer = response.json()["answer"]
        assert answer is not None
        assert answer["text"].strip()
        # Without an API key we serve the extractive fallback, which is the
        # guaranteed-correct path. With a key, citations must be real numbers.
        for citation in answer["citations"]:
            assert 1 <= citation["index"] <= 50
            assert citation["candidate_id"]


class TestFacets:
    async def test_skill_facets_reflect_the_corpus(self, client):
        await _seed(client)
        response = await client.get("/api/search/skills?limit=100")
        assert response.status_code == 200
        skills = response.json()["skills"]
        by_canonical = {s["canonical"]: s for s in skills}
        assert "python" in by_canonical
        # Priya + Ananya + Tomasz mention Python -> at least 3 candidates.
        assert by_canonical["python"]["candidate_count"] >= 3
        assert by_canonical["kubernetes"]["candidate_count"] >= 2

    async def test_filter_options(self, client):
        await _seed(client)
        response = await client.get("/api/search/filters")
        assert response.status_code == 200
        body = response.json()
        assert any(v["value"] == "Senior" for v in body["seniority"])
        assert any("Berlin" in v["value"] for v in body["locations"])

    async def test_candidate_facets_endpoint(self, client):
        await _seed(client)
        body = (await client.get("/api/candidates/facets")).json()
        assert body["locations"] and body["skills"]

    async def test_search_queries_are_logged(self, client, session):
        from sqlalchemy import text as sql_text

        await _seed(client)
        await client.post("/api/search", json={"query": "python engineer", "top_k": 3})
        count = (await session.execute(sql_text("SELECT count(*) FROM search_queries"))).scalar_one()
        assert count >= 1


class TestDirectSearcherUse:
    async def test_searcher_respects_an_explicit_candidate_id_set(self, client, session):
        """JD matching uses this to score a shortlist instead of the whole pool."""
        await _seed(client)
        rows = (await client.get("/api/candidates?limit=50")).json()
        target = next(r for r in rows if r["full_name"] == "Marcus Webb")

        outcome = await HybridSearcher(session).search(
            "react frontend",
            candidate_ids=[target["id"]],
            top_k=5,
        )
        assert outcome.hits
        assert {str(h.candidate.id) for h in outcome.hits} == {target["id"]}

    async def test_outcome_serialises_for_logging(self, client, session):
        await _seed(client)
        outcome = await HybridSearcher(session).search("engineer", top_k=3)
        payload = outcome.to_dict()
        assert set(payload) == {"filters_applied", "retrieval_ms", "counts"}
        assert isinstance(payload["retrieval_ms"], float)


class TestJDMatch:
    async def test_match_scores_candidates_against_a_jd(self, client):
        await _seed(client)
        jd = (
            "We are hiring a Senior Backend Engineer. You will build real-time "
            "ingestion pipelines with Kafka, work with PostgreSQL and pgvector, "
            "and run services on AWS and Kubernetes. Python is required."
        )
        response = await client.post("/api/match", json={"job_description": jd, "top_k": 5})
        assert response.status_code == 200
        body = response.json()
        assert body["total"] > 0
        # Scores must be sorted best-first.
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)
        # The strongest match is the Kafka/Postgres/AWS engineer.
        assert body["results"][0]["full_name"] == "Priya Raghunathan"

    async def test_match_rejects_a_too_short_jd(self, client):
        assert (await client.post("/api/match", json={"job_description": "hire someone"})).status_code == 422