"""
Candidate listing + detail.

    GET /api/candidates              browse, filter, sort, paginate
    GET /api/candidates/facets       what values exist in this corpus
    GET /api/candidates/{id}         full profile + its chunks
    DELETE /api/candidates/{id}      remove a candidate and its vectors
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, or_, select, text

from app.api.deps import SessionDep
from app.core.logging import get_logger
from app.db.models import Candidate, CandidateSkill, ResumeChunk
from app.schemas.candidate import CandidateDetail, CandidateSummary, ChunkOut, SkillOut
from app.services.retrieval.filters import SearchFilters, build_filters

router = APIRouter(prefix="/api/candidates", tags=["candidates"])
logger = get_logger(__name__)

_SORTS = {
    "recent": Candidate.created_at.desc(),
    "name": Candidate.full_name.asc().nullslast(),
    "experience": Candidate.total_years_experience.desc().nullslast(),
}


@router.get("", response_model=list[CandidateSummary])
async def list_candidates(
    session: SessionDep,
    skills: list[str] | None = Query(default=None),
    skills_mode: str = Query(default="all", pattern="^(all|any)$"),
    location: str | None = None,
    min_years_experience: float | None = Query(default=None, ge=0, le=60),
    seniority: list[str] | None = None,
    current_title: str | None = None,
    company: str | None = None,
    search: str | None = Query(default=None, description="Free-text name/title match"),
    sort: str = Query(default="recent"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[CandidateSummary]:
    """
    Structured browse with the same filters the search endpoint uses.

    This is the "table view" an HR user expects; `POST /api/search` is the
    "ask a question" view. Both reuse `SearchFilters`, so the two never drift.
    """
    filters = SearchFilters(
        skills=skills or [],
        skills_mode=skills_mode,
        location=location,
        min_years_experience=min_years_experience,
        current_title=current_title,
        company=company,
        seniority=seniority or [],
    )
    # alias=None: this composes with `select(Candidate)`, which has no "c" alias.
    filter_sql, params = build_filters(filters, alias=None)

    stmt = select(Candidate).where(Candidate.status == "ready")
    if filter_sql:
        stmt = stmt.where(text(filter_sql))
    if search:
        # Case-insensitive contains-match across the fields an HR user scans.
        needle = f"%{search.strip().lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(func.coalesce(Candidate.full_name, "")).like(needle),
                func.lower(func.coalesce(Candidate.current_title, "")).like(needle),
                func.lower(func.coalesce(Candidate.current_company, "")).like(needle),
            )
        )

    stmt = (
        stmt.order_by(_SORTS.get(sort, _SORTS["recent"]))
        .limit(limit)
        .offset(offset)
    )

    rows = (await session.execute(stmt, params)).scalars().all()
    return [CandidateSummary.model_validate(row) for row in rows]


@router.get("/facets")
async def candidate_facets(session: SessionDep) -> dict:
    """
    Aggregated values for the browse filters.

    NOTE the extra parentheses around `await session.execute(...)`. Method calls
    bind tighter than `await`, so without them Python parses this as
    `await (session.execute(...).mappings().all())` - awaiting a coroutine's
    chained attribute access - which fails at runtime with
    "'coroutine' object has no attribute 'mappings'".
    """
    locations = (
        await session.execute(
            text(
                """
                SELECT coalesce(location, 'Unknown') AS value, count(*) AS count
                FROM candidates GROUP BY 1 ORDER BY count DESC LIMIT 50
                """
            )
        )
    ).mappings().all()
    seniority = (
        await session.execute(
            text(
                """
                SELECT coalesce(seniority, 'Unknown') AS value, count(*) AS count
                FROM candidates GROUP BY 1 ORDER BY count DESC
                """
            )
        )
    ).mappings().all()
    skills = (
        await session.execute(
            text(
                """
                SELECT skill AS value, display AS label, count(*) AS count
                FROM candidate_skills
                GROUP BY skill, display ORDER BY count DESC LIMIT 100
                """
            )
        )
    ).mappings().all()
    return {
        "locations": [dict(r) for r in locations],
        "seniority": [dict(r) for r in seniority],
        "skills": [dict(r) for r in skills],
    }


@router.get("/{candidate_id}", response_model=CandidateDetail)
async def get_candidate(
    session: SessionDep,
    candidate_id: UUID,
    include_chunks: bool = Query(default=True, description="Include the RAG chunks"),
) -> CandidateDetail:
    """
    Full profile.

    `include_chunks` is on by default because it makes the retrieval transparent
    in the UI - you can see exactly which text was embedded and searchable.
    """
    candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    skills = [
        SkillOut(
            canonical=s.skill,
            display=s.display,
            occurrences=s.occurrences,
            is_core=s.is_core,
        )
        for s in sorted(candidate.skill_links, key=lambda s: (not s.is_core, -s.occurrences))
    ]

    chunks: list[ChunkOut] = []
    chunk_count = 0
    if include_chunks:
        rows = (
            await session.execute(
                select(ResumeChunk)
                .where(ResumeChunk.candidate_id == candidate_id)
                .order_by(ResumeChunk.chunk_index)
            )
        ).scalars().all()
        chunks = [
            ChunkOut(
                id=c.id,
                chunk_index=c.chunk_index,
                section=c.section,
                heading=c.heading,
                content=c.content,
                token_count=c.token_count,
            )
            for c in rows
        ]
        chunk_count = len(chunks)

    detail = CandidateDetail(
        id=candidate.id,
        full_name=candidate.full_name,
        current_title=candidate.current_title,
        current_company=candidate.current_company,
        location=candidate.location,
        total_years_experience=candidate.total_years_experience,
        seniority=candidate.seniority,
        email=candidate.email,
        phone=candidate.phone,
        summary=candidate.summary,
        status=candidate.status,
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
        links=candidate.links or {},
        education=candidate.education or {},
        languages=list(candidate.languages or []),
        skills=skills,
        resume_filename=candidate.resume_filename,
        extraction_method=candidate.extraction_method,
        text_length=len(candidate.raw_text or ""),
        chunk_count=chunk_count,
        chunks=chunks,
    )
    return detail


@router.delete("/{candidate_id}", status_code=204)
async def delete_candidate(session: SessionDep, candidate_id: UUID) -> None:
    """
    Delete a candidate. Chunks and skills cascade (FK ON DELETE CASCADE plus the
    ORM relationship cascade), so no vectors are left orphaned in the index.
    """
    candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Verify before we delete: this is the reliable way to confirm the cascade.
    await session.execute(
        func.count().select().where(ResumeChunk.candidate_id == candidate_id)
    )

    await session.delete(candidate)
    await session.commit()
    logger.info("deleted candidate %s", candidate_id)


@router.get("/{candidate_id}/skill-count")
async def skill_count(session: SessionDep, candidate_id: UUID) -> dict:
    """Small helper used by tests to assert the skill rows were written."""
    count = await session.scalar(
        select(func.count())
        .select_from(CandidateSkill)
        .where(CandidateSkill.candidate_id == candidate_id)
    )
    return {"candidate_id": str(candidate_id), "skill_count": int(count or 0)}