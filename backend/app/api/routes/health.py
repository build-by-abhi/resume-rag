"""
Health + stats endpoint.

Also the fastest way to check "is my setup actually wired up?":
if `embedding_provider` and `candidate_count` look right, ingestion works.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.deps import SessionDep
from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Candidate, ResumeChunk
from app.schemas.common import HealthResponse, LLMStatus
from app.services.embeddings import get_embedder
from app.services.llm.client import llm_available, llm_status
from app.services.vectordb.pgvector_store import PgVectorStore

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health(session: SessionDep) -> HealthResponse:
    started = time.perf_counter()
    db_state = "down"
    candidate_count = 0
    chunk_count = 0
    details: dict = {}

    try:
        store = PgVectorStore(session)
        candidate_count = await store.count_candidates()
        chunk_count = await store.count_chunks()
        db_state = "ok"

        # Cheap consistency check: candidates with no chunks are invisible to
        # search, which is the single most confusing failure mode.
        orphaned = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Candidate)
                    .outerjoin(ResumeChunk, ResumeChunk.candidate_id == Candidate.id)
                    .where(ResumeChunk.id.is_(None))
                )
            ).scalar_one()
        )
        if orphaned:
            details["candidates_without_chunks"] = orphaned
    except Exception as exc:
        db_state = "down"
        details["error"] = str(exc)[:300]
        logger.warning("health check: database unreachable (%s)", exc)

    try:
        embedder = get_embedder()
        details["embedding_model"] = getattr(embedder, "model", None) or settings.embedding_model
    except Exception as exc:
        details["embedding_error"] = str(exc)[:200]

    details["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)

    return HealthResponse(
        status="ok" if db_state == "ok" else "degraded",
        environment=settings.environment,
        database=db_state,
        embedding_provider=settings.embedding_provider,
        embedding_dim=settings.embedding_dim,
        llm_enabled=llm_available(),
        llm_status=LLMStatus(**llm_status()),
        candidate_count=candidate_count,
        chunk_count=chunk_count,
        details=details,
    )


@router.get("/stats", tags=["health"])
async def stats(session: SessionDep) -> dict:
    """Counts used by the UI header and useful in a demo screenshot."""
    store = PgVectorStore(session)
    total = await store.count_candidates()
    chunks = await store.count_chunks()
    avg_chunks = round(chunks / total, 1) if total else 0
    return {
        "candidates": total,
        "chunks": chunks,
        "avg_chunks_per_candidate": avg_chunks,
        "embedding_provider": settings.embedding_provider,
        "llm_enabled": llm_available(),
    }