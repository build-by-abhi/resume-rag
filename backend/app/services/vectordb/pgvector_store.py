"""
pgvector access layer.

Everything that speaks SQL against `resume_chunks` / `candidates` lives here so
the retrieval logic stays readable and the SQL is easy to review in one place.

Distance vs similarity
----------------------
pgvector's `<=>` is COSINE DISTANCE, which runs 0..2 (lower = better). We
convert to a 0..1 similarity so numbers are comparable across the two search
legs when we fuse them:

    similarity = 1 - (distance / 2)      ->  cosine similarity in 0..1

How the query vector is bound
-----------------------------
The vector goes in as a `bindparam` typed as `pgvector.Vector`, holding a plain
Python list. Two earlier approaches do NOT work and are worth knowing about:

* `text("CAST(:qv AS vector)")` - the bound string never reaches pgvector's bind
  processor, so it is treated as an array-like and raises
  "expected ndim to be 1".
* `.params(...)` on a statement that mixes `text()` fragments - the parameters
  silently end up empty and the query fails at execution.

A typed `bindparam` plus an explicit params dict passed to `execute()` is the
predictable path, and it is the only one that keeps user filter values as bound
parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Float, bindparam, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Candidate, ResumeChunk

logger = get_logger(__name__)

#: Cosine distance is at most 2.0, so this maps distance -> similarity.
_DISTANCE_SCALE = 2.0


def _vector_param(vector: Sequence[float], name: str, dim: int):
    """A typed bind parameter carrying the query vector."""
    return bindparam(
        name,
        value=[float(v) for v in vector],
        type_=Vector(dim),
    )


def cosine_similarity_expression(vector_param):
    """
    SQL expression: 1 - (embedding <=> :qv) / 2  ->  0..1, higher is better.

    `return_type=Float` is REQUIRED. Without it the `<=>` operator has an
    unknown result type, and SQLAlchemy then propagates the column's VECTOR type
    onto the surrounding arithmetic literals (`1.0`, `2.0`). Those float
    bind parameters then get pgvector's bind processor and every query dies with
    "expected ndim to be 1".
    """
    distance = ResumeChunk.embedding.op("<=>", return_type=Float)(vector_param)
    return (1.0 - distance / _DISTANCE_SCALE).label("vector_score")


def candidate_similarity_expression(vector_param):
    """Same conversion for the candidate-level profile vector."""
    distance = Candidate.profile_embedding.op("<=>", return_type=Float)(vector_param)
    return (1.0 - distance / _DISTANCE_SCALE).label("vector_score")


class PgVectorStore:
    """Thin data-access object around the two pgvector-backed tables."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # -- writes -------------------------------------------------------------

    async def delete_chunks_for(self, candidate_id) -> int:
        result = await self.session.execute(
            delete(ResumeChunk).where(ResumeChunk.candidate_id == candidate_id)
        )
        return result.rowcount or 0

    async def count_chunks(self) -> int:
        return int((await self.session.scalar(select(func.count()).select_from(ResumeChunk))) or 0)

    async def count_candidates(self) -> int:
        return int((await self.session.scalar(select(func.count()).select_from(Candidate))) or 0)

    # -- reads --------------------------------------------------------------

    async def semantic_chunk_search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 50,
        filter_sql: str | None = None,
        filter_params: dict[str, Any] | None = None,
        candidate_ids: Sequence[str] | None = None,
    ) -> list[dict]:
        """
        Vector leg: nearest chunks by cosine distance.

        `filter_sql` is a SQL fragment built by retrieval/filters.py. It is
        internal code; user values reach it only through bound parameters in
        `filter_params`.
        """
        dim = ResumeChunk.embedding.type.dim
        vector_param = _vector_param(query_vector, "qv", dim)

        stmt = select(
            ResumeChunk.id,
            ResumeChunk.candidate_id,
            ResumeChunk.chunk_index,
            ResumeChunk.section,
            ResumeChunk.heading,
            ResumeChunk.content,
            cosine_similarity_expression(vector_param),
        ).where(ResumeChunk.embedding.is_not(None))

        if candidate_ids is not None:
            if not candidate_ids:
                return []
            stmt = stmt.where(ResumeChunk.candidate_id.in_(list(candidate_ids)))
        if filter_sql:
            stmt = stmt.where(text(filter_sql))

        # `<=>` ASC == most similar first. This is exactly the access pattern the
        # HNSW index accelerates.
        stmt = stmt.order_by(
            ResumeChunk.embedding.op("<=>", return_type=Float)(
                _vector_param(query_vector, "qv", dim)
            )
        ).limit(limit)

        rows = (
            await self.session.execute(stmt, {"qv": list(query_vector), **(filter_params or {})})
        ).mappings().all()
        return [dict(row) for row in rows]

    async def keyword_chunk_search(
        self,
        query_text: str,
        *,
        limit: int = 50,
        filter_sql: str | None = None,
        filter_params: dict[str, Any] | None = None,
        candidate_ids: Sequence[str] | None = None,
    ) -> list[dict]:
        """
        Keyword leg: Postgres full-text search (BM25-ish via ts_rank_cd).

        `websearch_to_tsquery` accepts the loose query an HR person types
        ("senior python dev", "C++", quoted phrases, OR) without us writing a
        query parser, and the 'english' config stems "engineers" -> "engineer".
        """
        tsquery = func.websearch_to_tsquery("english", query_text)
        rank = func.ts_rank_cd(ResumeChunk.search_tsv, tsquery).label("keyword_score")

        stmt = (
            select(
                ResumeChunk.id,
                ResumeChunk.candidate_id,
                ResumeChunk.chunk_index,
                ResumeChunk.section,
                ResumeChunk.heading,
                ResumeChunk.content,
                rank,
            )
            .where(ResumeChunk.search_tsv.is_not(None))
            .where(tsquery.op("@@", return_type=Boolean)(ResumeChunk.search_tsv))
        )

        if candidate_ids is not None:
            if not candidate_ids:
                return []
            stmt = stmt.where(ResumeChunk.candidate_id.in_(list(candidate_ids)))
        if filter_sql:
            stmt = stmt.where(text(filter_sql))

        stmt = stmt.order_by(rank.desc()).limit(limit)
        rows = (await self.session.execute(stmt, dict(filter_params or {}))).mappings().all()
        return [dict(row) for row in rows]

    async def candidate_search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 50,
        filter_sql: str | None = None,
        filter_params: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Candidate-level vector search on the whole-profile embedding."""
        dim = Candidate.profile_embedding.type.dim
        vector_param = _vector_param(query_vector, "qv", dim)

        stmt = select(
            Candidate.id,
            Candidate.full_name,
            Candidate.email,
            Candidate.phone,
            Candidate.location,
            Candidate.current_title,
            Candidate.current_company,
            Candidate.total_years_experience,
            Candidate.seniority,
            Candidate.summary,
            candidate_similarity_expression(vector_param),
        ).where(Candidate.profile_embedding.is_not(None))

        if filter_sql:
            stmt = stmt.where(text(filter_sql))

        stmt = stmt.order_by(
            Candidate.profile_embedding.op("<=>", return_type=Float)(
                _vector_param(query_vector, "qv", dim)
            )
        ).limit(limit)

        rows = (
            await self.session.execute(stmt, {"qv": list(query_vector), **(filter_params or {})})
        ).mappings().all()
        return [dict(row) for row in rows]

    async def get_candidates(self, candidate_ids: Sequence[str]) -> list[Candidate]:
        if not candidate_ids:
            return []
        return list(
            (
                await self.session.execute(
                    select(Candidate).where(Candidate.id.in_(list(candidate_ids)))
                )
            )
            .scalars()
            .all()
        )


def get_store(session: AsyncSession) -> PgVectorStore:
    return PgVectorStore(session)


__all__ = [
    "PgVectorStore",
    "get_store",
    "cosine_similarity_expression",
    "candidate_similarity_expression",
]