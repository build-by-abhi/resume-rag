"""
Search endpoint (Phases 2 + 3).

GET  /api/search/filters    filter options for the UI dropdowns
GET  /api/search/skills     skill facets with candidate counts
POST /api/search            hybrid search + optional LLM answer
POST /api/match             JD -> candidate scorecards (Phase 4)

The single search call does three things in order:

  1. **structured filters**  exact match on columns (fast, certain)
  2. **hybrid retrieval**    BM25 + vector, fused with RRF, optionally reranked
  3. **grounded answer**     LLM writes a cited answer, or an extractive digest

Every step degrades gracefully: no filters -> pure retrieval; no LLM key ->
extractive answer; no matches -> a clear message instead of an error.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from sqlalchemy import text as sql_text

from app.api.deps import SessionDep
from app.core.logging import get_logger
from app.db.models import SearchQuery
from app.schemas.search import (
    AnswerOut,
    CandidateResult,
    CitationOut,
    JDMatchRequest,
    JDMatchResponse,
    JDMatchResult,
    SearchRequest,
    SearchResponse,
)
from app.services.llm.client import NullLLMClient, get_llm_client
from app.services.retrieval.answer import answer_generator, group_hits_for_display
from app.services.retrieval.filters import SKILL_FACET_SQL, SearchFilters
from app.services.retrieval.hybrid import HybridSearcher

router = APIRouter(prefix="/api", tags=["search"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Facets for the UI
# ---------------------------------------------------------------------------
@router.get("/search/skills")
async def skill_facets(session: SessionDep, limit: int = 60) -> dict:
    """
    Which skills exist in this corpus, and how many candidates have each.

    Driving the filter dropdown from the data (rather than a hardcoded list)
    means the UI always reflects the talent pool you actually have.
    """
    rows = (
        await session.execute(sql_text(SKILL_FACET_SQL), {"_facet_limit": limit})
    ).mappings().all()
    return {
        "skills": [
            {
                "canonical": row["skill"],
                "display": row["display"],
                "candidate_count": row["candidate_count"],
                "is_core": row["is_core"],
            }
            for row in rows
        ]
    }


@router.get("/search/filters")
async def filter_options(session: SessionDep) -> dict:
    """Distinct values for the dropdowns, so nothing is hardcoded in the UI."""
    # NB: the parentheses around `await session.execute(...)` are required -
    # `.mappings()` binds tighter than `await`.
    rows = (
        await session.execute(
            sql_text(
                """
                SELECT
                  coalesce(seniority, 'Unknown')  AS seniority,
                  count(*)                        AS count
                FROM candidates
                GROUP BY seniority
                ORDER BY count DESC
                """
            )
        )
    ).mappings().all()
    locations = (
        await session.execute(
            sql_text(
                """
                SELECT coalesce(split_part(location, ',', 1), '') AS city, count(*) AS count
                FROM candidates
                WHERE location IS NOT NULL AND location <> ''
                GROUP BY 1
                ORDER BY count DESC, city
                LIMIT 50
                """
            )
        )
    ).mappings().all()
    return {
        "seniority": [{"value": r["seniority"], "count": r["count"]} for r in rows],
        "locations": [{"value": r["city"], "count": r["count"]} for r in locations if r["city"]],
        "education_levels": [
            "PhD", "MTech", "MS", "MBA", "BSc", "Associate", "HighSchool", "Diploma",
        ],
        "skills_mode_options": ["all", "any"],
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@router.post("/search", response_model=SearchResponse)
async def search(session: SessionDep, request: SearchRequest) -> SearchResponse:
    started = time.perf_counter()

    if not request.has_text_or_filters:
        raise HTTPException(
            status_code=422,
            detail="Provide a question in 'query' or at least one filter.",
        )

    filters = SearchFilters(
        skills=request.skills,
        skills_mode=request.skills_mode,
        location=request.location,
        min_years_experience=request.min_years_experience,
        max_years_experience=request.max_years_experience,
        current_title=request.current_title,
        company=request.company,
        seniority=request.seniority,
        languages=request.languages,
        education_level=request.education_level,
        has_email=request.has_email,
        has_phone=request.has_phone,
    )

    searcher = HybridSearcher(session)

    if request.query:
        # Full hybrid path: text + filters.
        outcome = await searcher.search(
            request.query,
            filters=filters,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
        )
    else:
        # Filters-only mode ("show me every Senior person in Berlin").
        # There is no ranking signal without a query, so we list the matching
        # candidates ordered by experience instead of pretending to rank them.
        from app.services.retrieval.hybrid import SearchOutcome

        started_at = time.perf_counter()
        hits = await _browse_candidates(session, filters, request.top_k)
        outcome = SearchOutcome(
            hits=hits,
            filters_applied=filters.to_dict(),
            retrieval_ms=(time.perf_counter() - started_at) * 1000,
            counts={"mode": "filters_only", "candidates_returned": len(hits)},
        )

    # ---- answer -------------------------------------------------------
    answer_out: AnswerOut | None = None
    if request.generate_answer and outcome.hits:
        answer = await answer_generator.answer(
            request.query or _describe_filters(filters),
            outcome,
        )
        answer_out = AnswerOut(
            text=answer.text,
            used_llm=answer.used_llm,
            model=answer.model,
            latency_ms=round(answer.latency_ms, 1) if answer.latency_ms else None,
            insufficient_evidence=answer.insufficient_evidence,
            had_invalid_citations=answer.had_invalid_citations,
            citations=[
                CitationOut(
                    index=c.index,
                    candidate_id=c.candidate_id,
                    candidate_name=c.candidate_name,
                    section=c.section,
                    heading=c.heading,
                    excerpt=c.excerpt,
                )
                for c in answer.citations
            ],
        )

    total_ms = (time.perf_counter() - started) * 1000

    # ---- audit log -----------------------------------------------------
    session.add(
        SearchQuery(
            query_text=request.query or "[filters only]",
            filters=filters.to_dict(),
            result_count=len(outcome.hits),
            top_candidate_ids=[str(h.candidate.id) for h in outcome.hits[:10]],
            retrieval_ms=outcome.retrieval_ms,
            llm_ms=(answer.latency_ms if (answer_out and answer_out.used_llm) else None),
            used_llm=bool(answer_out and answer_out.used_llm),
        )
    )
    await session.commit()

    return SearchResponse(
        query=request.query,
        total=len(outcome.hits),
        results=_to_results(outcome),
        answer=answer_out,
        filters_applied=filters.to_dict(),
        retrieval_ms=round(outcome.retrieval_ms, 1),
        total_ms=round(total_ms, 1),
        diagnostics=outcome.to_dict()["counts"],
    )


@router.post("/match", response_model=JDMatchResponse)
async def match_job_description(session: SessionDep, request: JDMatchRequest) -> JDMatchResponse:
    """
    Phase 4: JD -> candidate match.

    Retrieval reuses the hybrid searcher (the JD becomes the query), then an
    optional LLM scores each candidate against the JD with citations.
    """
    started = time.perf_counter()
    searcher = HybridSearcher(session)
    outcome = await searcher.search(request.job_description, top_k=request.top_k)

    if not outcome.hits:
        return JDMatchResponse(
            job_description_excerpt=request.job_description[:400],
            total=0,
            retrieval_ms=outcome.retrieval_ms,
            total_ms=(time.perf_counter() - started) * 1000,
        )

    client = get_llm_client()
    used_llm = not isinstance(client, NullLLMClient) and request.generate_analysis

    from app.prompts.templates import JD_MATCH_SCHEMA_HINT, build_jd_match_prompt

    results: list[JDMatchResult] = []
    for hit in outcome.hits:
        blocks = [
            (str(i + 1), chunk.content) for i, chunk in enumerate(hit.citations[:4])
        ]
        scorecard: dict = {}
        if used_llm:
            system, user = build_jd_match_prompt(request.job_description, blocks)
            payload = await client.complete_json(f"{system}\n\n{JD_MATCH_SCHEMA_HINT}", user)
            if isinstance(payload, dict):
                scorecard = payload

        fallback_score = round(min(99.0, hit.score * 100), 1)
        results.append(
            JDMatchResult(
                candidate_id=hit.candidate.id,
                full_name=hit.candidate.full_name,
                current_title=hit.candidate.current_title,
                score=float(scorecard.get("score", fallback_score) or fallback_score),
                skills_match=_as_float(scorecard.get("skills_match")),
                experience_match=_as_float(scorecard.get("experience_match")),
                seniority_match=_as_float(scorecard.get("seniority_match")),
                missing_critical_skills=[
                    str(s) for s in (scorecard.get("missing_critical_skills") or [])
                ][:15],
                strengths=[s for s in (scorecard.get("strengths") or []) if isinstance(s, dict)],
                summary=scorecard.get("summary"),
                evidence=[c.to_dict() for c in hit.citations],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    results = [r for r in results if r.score >= request.min_score]

    return JDMatchResponse(
        job_description_excerpt=request.job_description[:400],
        total=len(results),
        results=results,
        used_llm=used_llm,
        retrieval_ms=round(outcome.retrieval_ms, 1),
        total_ms=round((time.perf_counter() - started) * 1000, 1),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_results(outcome) -> list:
    """
    Convert hits into `CandidateResult`.

    `group_hits_for_display` already returns plain dicts shaped like the schema,
    so Pydantic does the coercion - including nested `evidence` -> EvidenceOut.
    """
    return [CandidateResult(**payload) for payload in group_hits_for_display(outcome.hits)]


async def _browse_candidates(session, filters: SearchFilters, top_k: int) -> list:
    """
    Filters-only mode ("show me all Senior people in Berlin").

    No query text means no ranking signal, so we return the matching candidates
    ordered by a sensible default (most experienced first) instead of pretending
    to rank them by relevance.
    """
    from sqlalchemy import text as sql_text

    from app.services.retrieval.filters import build_filters

    sql, params = build_filters(filters)
    if not sql:
        stmt = sql_text(
            "SELECT c.id::text FROM candidates c "
            "ORDER BY c.total_years_experience DESC NULLS LAST, c.created_at DESC LIMIT :_k"
        )
        params = {"_k": top_k}
    else:
        stmt = sql_text(
            f"SELECT c.id::text FROM candidates c WHERE {sql} "
            "ORDER BY c.total_years_experience DESC NULLS LAST, c.created_at DESC LIMIT :_k"
        )
        params = {**params, "_k": top_k}

    ids = [str(r) for r in (await session.execute(stmt, params)).scalars().all()]
    if not ids:
        return []

    from sqlalchemy import select

    from app.db.models import Candidate
    from app.services.retrieval.hybrid import CandidateHit

    rows_candidates = await session.execute(select(Candidate).where(Candidate.id.in_(ids)))
    candidates = {str(c.id): c for c in rows_candidates.scalars().all()}

    hits: list[CandidateHit] = []
    for index, cid in enumerate(ids):
        candidate = candidates.get(cid)
        if candidate is None:
            continue
        # No ranking signal -> use position order as the score.
        hits.append(
            CandidateHit(
                candidate=candidate,
                score=max(0.0, 1.0 - index / max(len(ids), 1)),
                best_chunk=None,
                supporting_chunks=[],
                matched_skills=[s.skill for s in getattr(candidate, "skill_links", []) or []],
            )
        )
    return hits


def _describe_filters(filters: SearchFilters) -> str:
    """Human-readable query text for the answer prompt in filters-only mode."""
    bits: list[str] = []
    if filters.skills:
        bits.append(f"skills {filters.skills_mode} [{', '.join(filters.skills)}]")
    if filters.location:
        bits.append(f"location '{filters.location}'")
    if filters.min_years_experience is not None:
        bits.append(f"at least {filters.min_years_experience:g} years experience")
    if filters.current_title:
        bits.append(f"title '{filters.current_title}'")
    if filters.seniority:
        bits.append(f"seniority {filters.seniority}")
    return "candidates matching " + (", ".join(bits) if bits else "all filters")


__all__ = ["router"]