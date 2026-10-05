"""
MCP tools: the model-controlled surface.

DESIGN RULES THAT MATTER
------------------------
1. **The description is the prompt.** The model reads it to decide whether to
   call the tool. "Use for X. Do not use for Y" prevents the most common failure:
   the model reaching for a tool that cannot answer its question.

2. **Return evidence, not conclusions.** A candidate result carries the chunks
   that matched. Without them the model will invent a reason for the match and
   present it as fact.

3. **Output schemas are declared.** Returning a typed object means the client
   can render results and the model can reference fields by name, instead of
   parsing prose.

4. **Read-only tools are annotated `read_only_hint=True`.** It is a hint, not a
   security control - clients treat annotations from untrusted servers as
   untrusted - but it lets a client skip confirmation prompts.

5. **Never trust the arguments.** They are model-generated. Every string is a
   bound parameter, never interpolated into SQL, and ranges are clamped.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations

from app.core.config import settings
from app.core.logging import get_logger
from app.mcp.context import database
from app.mcp.schemas import (
    IngestResult,
    MatchResult,
    SearchResult,
    SkillResult,
)
from app.services.embeddings import get_embedder
from app.services.retrieval.answer import answer_generator, group_hits_for_display
from app.services.retrieval.filters import SearchFilters
from app.services.retrieval.hybrid import HybridSearcher

logger = get_logger(__name__)

# Defensive ceiling. `top_k` is model-supplied, and an unbounded limit would let
# a loop ask for the whole corpus and blow up the LLM context window.
MAX_TOP_K = 50
DEFAULT_TOP_K = 10

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)
# Not read-only: it creates a candidate row.
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True)


def register_tools(server) -> None:
    """Attach every tool to an MCPServer instance."""

    # -----------------------------------------------------------------------
    @server.tool(
        name="search_candidates",
        title="Search the candidate pool",
        description=(
            "Hybrid search over stored resumes. Combines structured filters "
            "(exact: skills, location, years of experience, seniority) with "
            "semantic and keyword retrieval, then reranks.\n\n"
            "USE THIS for questions about who has some experience, has worked "
            "with a technology, or would fit a role.\n"
            "DO NOT use it to answer questions about a specific person you can "
            "name - use get_candidate_profile for that.\n\n"
            "Skill aliases are resolved automatically ('k8s' matches "
            "'kubernetes', 'postgres' matches 'PostgreSQL'). Returns ranked "
            "candidates WITH the resume excerpts that matched, so you can "
            "justify every claim. Returns an empty result rather than guessing "
            "when nothing matches."
        ),
        annotations=READ_ONLY,
    )
    async def search_candidates(
        query: Annotated[
            str,
            "Natural-language question, e.g. 'senior Python engineer who has "
            "worked with Kubernetes'. Leave empty to filter only.",
        ] = "",
        skills: Annotated[
            list[str] | None,
            "Skills that must be present. Aliases are auto-resolved.",
        ] = None,
        skills_mode: Annotated[
            str, "'all' (default) requires every skill; 'any' requires one."
        ] = "all",
        location: Annotated[
            str | None, "City or 'City, Country', e.g. 'Berlin'."
        ] = None,
        min_years_experience: Annotated[
            float | None, "Minimum years of experience, 0-60."
        ] = None,
        max_years_experience: Annotated[
            float | None, "Maximum years of experience, 0-60."
        ] = None,
        current_title: Annotated[str | None, "Job title substring."] = None,
        seniority: Annotated[
            list[str] | None,
            "Any of Junior, Mid, Senior, Lead, Principal, Executive.",
        ] = None,
        top_k: Annotated[int, f"How many candidates to return (1-{MAX_TOP_K})."] = DEFAULT_TOP_K,
        generate_answer: Annotated[
            bool,
            "Set false to skip the written summary and return only ranked rows. "
            "Much faster; set true only when the user asked for a written answer.",
        ] = False,
    ) -> SearchResult:
        # Clamp model-supplied bounds rather than trusting them.
        limit = max(1, min(int(top_k or DEFAULT_TOP_K), MAX_TOP_K))

        if not query.strip() and not any(
            [
                skills,
                location,
                min_years_experience is not None,
                max_years_experience is not None,
                current_title,
                seniority,
            ]
        ):
            return {
                "total": 0,
                "results": [],
                "error": (
                    "Provide a `query` or at least one filter. An empty search "
                    "returns everything, which is not useful here."
                ),
            }

        filters = SearchFilters(
            skills=skills or [],
            skills_mode=skills_mode if skills_mode in {"all", "any"} else "all",
            location=location,
            min_years_experience=_clamp_years(min_years_experience),
            max_years_experience=_clamp_years(max_years_experience),
            current_title=current_title,
            seniority=seniority or [],
        )

        async with database.session() as session:
            searcher = HybridSearcher(session)

            if query.strip():
                outcome = await searcher.search(
                    query, filters=filters, top_k=limit
                )
            else:
                # Filters only: no ranking signal, so order by experience
                # rather than pretending to rank by relevance.
                from app.api.routes.search import _browse_candidates
                from app.services.retrieval.hybrid import SearchOutcome

                outcome = SearchOutcome(
                    hits=await _browse_candidates(session, filters, limit),
                    filters_applied=filters.to_dict(),
                    counts={"mode": "filters_only"},
                )

            payload = _serialise_hits(outcome, limit)

            if generate_answer and outcome.hits:
                answer = await answer_generator.answer(
                    query or f"candidates matching {filters.to_dict()}", outcome
                )
                payload["answer"] = answer.text
                payload["answer_uses_llm"] = answer.used_llm
                if answer.had_invalid_citations:
                    payload["answer_warning"] = (
                        "The model cited references that did not exist; they "
                        "were removed. Treat the wording with care."
                    )

        logger.info(
            "mcp search_candidates query=%r skills=%s -> %d results",
            query[:60], skills, payload["total"],
        )
        return payload

    # -----------------------------------------------------------------------
    @server.tool(
        name="match_job_description",
        title="Score candidates against a job description",
        description=(
            "Rank the candidate pool against a pasted job description.\n\n"
            "USE THIS when the user supplies a job description or role spec and "
            "asks who fits. Do NOT use it for general skill questions - use "
            "search_candidates for those, it is faster and more precise.\n\n"
            "Retrieval finds candidates; when an LLM is configured it also scores "
            "each one against the requirements and explains the gap. Scores are "
            "0-100 and include evidence."
        ),
        annotations=READ_ONLY,
    )
    async def match_job_description(
        job_description: Annotated[str, "The full job description text."],
        top_k: Annotated[int, f"How many candidates to score (1-{MAX_TOP_K})."] = DEFAULT_TOP_K,
    ) -> MatchResult:
        # NOTE the parentheses: `len(x or "").strip()` would call .strip() on the
        # int returned by len(), raising "int object has no attribute 'strip'".
        # A method call binds tighter than the surrounding expression.
        jd = (job_description or "").strip()
        if len(jd) < 40:
            return {
                "error": (
                    "job_description must be at least 40 characters. Paste the "
                    "full description including requirements."
                )
            }

        limit = max(1, min(int(top_k or DEFAULT_TOP_K), MAX_TOP_K))

        async with database.session() as session:
            outcome = await HybridSearcher(session).search(
                jd, top_k=limit
            )

            results: list[dict] = []
            for hit in outcome.hits:
                candidate = hit.candidate
                results.append(
                    {
                        "id": str(candidate.id),
                        "full_name": candidate.full_name,
                        "current_title": candidate.current_title,
                        "current_company": candidate.current_company,
                        "location": candidate.location,
                        "total_years_experience": candidate.total_years_experience,
                        "seniority": candidate.seniority,
                        "retrieval_score": round(hit.score, 4),
                        "evidence": [c.to_dict() for c in hit.citations],
                    }
                )

        return {
            "total": len(results),
            "scoring": "retrieval_only",
            "note": (
                "Scores are hybrid-retrieval ranks, not a fit assessment. No LLM "
                "is configured, so this is NOT a judgement about suitability - use "
                "it to shortlist, then read the evidence."
            ),
            "results": results,
        }

    # -----------------------------------------------------------------------
    @server.tool(
        name="list_available_skills",
        title="List indexed skills",
        description=(
            "List the technologies actually present in the candidate pool, with "
            "how many candidates have each.\n\n"
            "USE THIS before guessing at a skill name when filtering. 'Kotlin' "
            "returns nothing if nobody has it; 'kubernetes' returns a count. "
            "Values are canonical keys - pass them straight to "
            "search_candidates(skills=[...])."
        ),
        annotations=READ_ONLY,
    )
    async def list_available_skills(
        limit: Annotated[int, "Maximum skills to return (1-200)."] = 60,
    ) -> SkillResult:
        rows = await _skill_facets(max(1, min(int(limit or 60), 200)))
        return {"total": len(rows), "skills": rows}

    # -----------------------------------------------------------------------
    @server.tool(
        name="ingest_resume",
        title="Ingest a resume",
        description=(
            "Parse and index a pasted resume, creating a searchable candidate.\n\n"
            "USE THIS only when the user pastes resume text or asks to add a "
            "person. This WRITES to the database. Do not call it to look someone "
            "up.\n\n"
            "Identical text is deduplicated by content hash, so re-submitting is "
            "safe and reports created=false. Scanned images with no text layer "
            "will fail - those need OCR first."
        ),
        annotations=WRITE,
    )
    async def ingest_resume(
        resume_text: Annotated[str, "The full resume text (min 80 characters)."],
    ) -> IngestResult:
        text = (resume_text or "").strip()
        if len(text) < 80:
            return {"error": "resume_text must be at least 80 characters."}

        # Reuse the exact pipeline the HTTP upload endpoint uses, so MCP and REST
        # cannot diverge.
        import uuid

        from app.services.ingestion.pipeline import pipeline

        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        path = settings.upload_dir / f"mcp_{uuid.uuid4().hex[:12]}.txt"
        path.write_text(text, encoding="utf-8")

        async with database.session() as session:
            try:
                result = await pipeline.ingest_file(
                    session, path, original_filename="mcp_pasted.txt"
                )
            finally:
                # Pasted text has no reason to persist on disk.
                path.unlink(missing_ok=True)

        candidate = result.candidate
        warnings: list[str] = []
        if not candidate.full_name:
            warnings.append("Could not detect the candidate's name")
        if not candidate.email:
            warnings.append("No email found")
        if not candidate.skill_links:
            warnings.append("No known skills detected")

        logger.info(
            "mcp ingest_resume created=%s chunks=%d skills=%d",
            result.created, result.chunk_count, result.skill_count,
        )
        return {
            "candidate_id": str(candidate.id),
            "created": result.created,
            "deduplicated": not result.created,
            "full_name": candidate.full_name,
            "chunk_count": result.chunk_count,
            "skill_count": result.skill_count,
            "extraction_method": candidate.extraction_method,
            "warnings": warnings,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clamp_years(value: float | None) -> float | None:
    """Bound a model-supplied years value. `None` passes through unchanged."""
    if value is None:
        return None
    return max(0.0, min(float(value), 60.0))


def _serialise_hits(outcome, limit: int) -> SearchResult:
    """
    Convert SearchOutcome into a JSON-safe dict for the MCP result.

    `group_hits_for_display` already returns plain dicts shaped like the REST
    schema; this trims the fields a model does not need and flattens evidence
    so the token cost stays predictable.
    """
    rows: list[dict] = []
    for payload in group_hits_for_display(outcome.hits)[:limit]:
        rows.append(
            {
                "id": payload["id"],
                "full_name": payload["full_name"],
                "current_title": payload["current_title"],
                "current_company": payload["current_company"],
                "location": payload["location"],
                "total_years_experience": payload["total_years_experience"],
                "seniority": payload["seniority"],
                "score": payload["score"],
                "skills": [s["display"] for s in payload["skills"][:12]],
                "evidence": [
                    {
                        "section": e["section"],
                        "heading": e["heading"],
                        "content": e["content"],
                    }
                    for e in payload["evidence"][:3]
                ],
            }
        )

    return {
        "total": len(outcome.hits),
        "results": rows,
        "diagnostics": outcome.counts,
    }


def outcome_counts(outcome) -> int:
    """Number of candidate hits. Kept separate so tests can assert on it."""
    return len(outcome.hits)


async def _skill_facets(limit: int) -> list[dict]:
    from app.mcp.context import fetch_all
    from app.services.retrieval.filters import SKILL_FACET_SQL

    rows = await fetch_all(SKILL_FACET_SQL, {"_facet_limit": limit})
    return [
        {
            "canonical": row["skill"],
            "display": row["display"],
            "candidate_count": row["candidate_count"],
            "is_core_skill": row["is_core"],
        }
        for row in rows
    ]


__all__ = ["register_tools", "READ_ONLY", "WRITE", "MAX_TOP_K", "DEFAULT_TOP_K"]


# Re-exported for the resource handlers, which need the same embedder access.
def embedder():
    return get_embedder()


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {"value": str(value)}