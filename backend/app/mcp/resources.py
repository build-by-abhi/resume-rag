"""
MCP resources: the application-controlled context surface.

TOOLS VS RESOURCES - THE DISTINCTION THAT MATTERS
-------------------------------------------------
    Tool      the MODEL decides whether to call it
    Resource  the APP decides what to attach to the conversation

Putting a read-only profile behind a tool means the model may never ask for it.
Putting a mutating operation behind a resource means the model fires it without
realising it did anything. Getting this backwards is the most common MCP design
error.

URI SCHEME
----------
    resume://candidates/{id}          one person's full profile + chunks
    resume://candidates               the whole pool as a summary list
    talent://facets/skills            skill histogram for the pool
    talent://corpus/stats             counts and configuration

The `resume://` prefix means these can also be used as MCP *resource
templates*, where the host discovers the shape by introspection rather than
hardcoding ids.
"""

from __future__ import annotations

import json
from typing import Annotated

from app.core.logging import get_logger
from app.mcp.context import fetch_all, fetch_one

logger = get_logger(__name__)


def register_resources(server) -> None:
    """Attach every resource and resource template."""

    # -----------------------------------------------------------------------
    # One candidate, in full.
    # -----------------------------------------------------------------------
    @server.resource(
        "resume://candidates/{candidate_id}",
        name="candidate_profile",
        title="Candidate profile",
        description=(
            "A single candidate: structured profile, skills, and every resume "
            "chunk that was embedded. Use this once you know WHO you are talking "
            "about - it is larger than a search result."
        ),
        mime_type="application/json",
    )
    async def candidate_profile(candidate_id: Annotated[str, "Candidate UUID."]) -> str:
        row = await fetch_one(
            """
            SELECT id, full_name, email, phone, location, current_title,
                   current_company, total_years_experience, seniority,
                   education, summary, links, languages, resume_filename,
                   extraction_method, created_at,
                   char_length(raw_text) AS text_length
            FROM candidates WHERE id = :cid
            """,
            {"cid": candidate_id},
        )
        if row is None:
            return json.dumps(
                {
                    "error": f"No candidate with id {candidate_id}.",
                    "hint": "Use search_candidates to find valid ids.",
                }
            )

        skills = await fetch_all(
            """
            SELECT display, occurrences, is_core
            FROM candidate_skills WHERE candidate_id = :cid
            ORDER BY is_core DESC, occurrences DESC, display
            """,
            {"cid": candidate_id},
        )
        chunks = await fetch_all(
            """
            SELECT chunk_index, section, heading, token_count, content
            FROM resume_chunks WHERE candidate_id = :cid
            ORDER BY chunk_index
            """,
            {"cid": candidate_id},
        )

        # `education` is a JSONB column and `created_at` a datetime; both need
        # normalising because the payload goes straight to json.dumps.
        row.setdefault("education", {})
        created = row.get("created_at")
        row["created_at"] = (
            created.isoformat() if hasattr(created, "isoformat") else created
        )

        return json.dumps(
            {
                **row,
                "skills": skills,
                "chunk_count": len(chunks),
                "chunks": chunks,
            },
            indent=2,
            default=str,
        )

    # -----------------------------------------------------------------------
    # The whole pool, as a summary. Useful as conversation grounding without
    # the caller having to search first.
    # -----------------------------------------------------------------------
    @server.resource(
        "resume://candidates",
        name="candidate_directory",
        title="Candidate directory",
        description=(
            "Every candidate in the pool as a one-line summary, ordered by "
            "experience. Use for broad questions like 'what seniority levels do "
            "we have' or to ground the model before it searches."
        ),
        mime_type="application/json",
    )
    async def candidate_directory() -> str:
        rows = await fetch_all(
            """
            SELECT id, full_name, current_title, current_company, location,
                   total_years_experience, seniority
            FROM candidates
            WHERE status = 'ready'
            ORDER BY total_years_experience DESC NULLS LAST, full_name
            """
        )
        return json.dumps(
            {"total": len(rows), "candidates": rows}, indent=2, default=str
        )

    # -----------------------------------------------------------------------
    # Skill histogram.
    # -----------------------------------------------------------------------
    @server.resource(
        "talent://facets/skills",
        name="skill_facets",
        title="Skill facets",
        description=(
            "Technology histogram for the pool: every indexed skill with a "
            "candidate count. Use to discover what filter values are valid "
            "before searching, so you never invent a skill that nobody has."
        ),
        mime_type="application/json",
    )
    async def skill_facets() -> str:
        rows = await fetch_all(
            """
            SELECT cs.skill AS canonical, min(cs.display) AS display,
                   count(DISTINCT cs.candidate_id) AS candidate_count,
                   bool_or(cs.is_core) AS is_core
            FROM candidate_skills cs
            JOIN candidates c ON c.id = cs.candidate_id
            GROUP BY cs.skill
            ORDER BY candidate_count DESC, display
            LIMIT 200
            """
        )
        return json.dumps({"total": len(rows), "skills": rows}, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Corpus statistics - how much is actually searchable?
    # -----------------------------------------------------------------------
    @server.resource(
        "talent://corpus/stats",
        name="corpus_stats",
        title="Corpus statistics",
        description=(
            "How many candidates, chunks and skills are indexed, plus the active "
            "embedding and LLM configuration. Read this first when a search "
            "returns nothing - it distinguishes 'no matches' from 'empty "
            "database' from 'LLM disabled'."
        ),
        mime_type="application/json",
    )
    async def corpus_stats() -> str:
        from app.core.config import settings
        from app.services.llm.client import llm_status

        row = await fetch_one(
            """
            SELECT
              (SELECT count(*) FROM candidates WHERE status = 'ready') AS candidates,
              (SELECT count(*) FROM resume_chunks)                    AS chunks,
              (SELECT count(*) FROM candidate_skills)                  AS skill_rows,
              (SELECT count(*) FROM resume_chunks WHERE embedding IS NULL)
                                                                     AS chunks_without_vector,
              (SELECT count(*) FROM search_queries)                   AS searches_logged
            """
        )
        status = llm_status()
        return json.dumps(
            {
                **(row or {}),
                "embedding_provider": settings.embedding_provider,
                "embedding_dim": settings.embedding_dim,
                "llm_enabled": status["configured"],
                "llm_reason": status["reason"],
            },
            indent=2,
            default=str,
        )


__all__ = ["register_resources"]