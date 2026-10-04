"""
Hybrid search: the heart of Phase 3.

PIPELINE
--------
    query
      │
      ├─ 1. STRUCTURED FILTERS  ──► list of candidate ids   (exact, indexed)
      │      (skip this step when there are no filters)
      │
      ├─ 2a. VECTOR search       ──► ranked chunk list      (semantic)
      ├─ 2b. KEYWORD search      ──► ranked chunk list      (BM25, exact terms)
      │
      ├─ 3. RRF FUSION           ──► one merged ranking
      │      (rank-based, so the two legs' incomparable scores never mix badly)
      │
      ├─ 4. RERANK (optional)    ──► cross-encoder scores the top N pairs
      │
      └─ 5. GROUP BY CANDIDATE   ──► best chunk per candidate + supporting chunks
             then answer via the LLM with those chunks as context

WHY HYBRID INSTEAD OF PURE VECTOR SEARCH
----------------------------------------
Each leg fails in a different, well-known way:

  * Pure vector: "C++" vs "C" collapses; "React" matches "Ruby"; a rare exact
    token like "5551234" or "Kubernetes CRD operator" is lost in the average.
  * Pure BM25: no synonyms ("ML" vs "machine learning" never meet), no notion of
    seniority, and it happily returns a 2-year resume for "senior engineer".

RRF (Reciprocal Rank Fusion) is the standard way to combine them because it uses
ONLY the ranks, not the raw scores -- a cosine of 0.71 and a BM25 score of 3.2
are not on the same scale, and trying to normalise them is guesswork.

    RRF(d) = Σ  1 / (k + rank_i(d))        k = 60 (default, from the paper)

Records that appear in both lists get roughly double the weight of records that
appear in only one -- which is exactly the behaviour we want for "senior Python
engineer": that resume should match both the semantic leg and the keyword leg.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings import get_embedder
from app.services.extraction.skills import expand_query_for_skills
from app.services.retrieval.filters import SearchFilters, resolve_filtered_candidate_ids
from app.services.retrieval.rerank import normalize_scores, rerank_chunks
from app.services.vectordb.pgvector_store import PgVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """One evidence item, with everything needed to cite it."""

    chunk_id: str
    candidate_id: str
    chunk_index: int
    section: str
    heading: str | None
    content: str
    score: float = 0.0
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "candidate_id": self.candidate_id,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "heading": self.heading,
            "content": self.content,
            "score": round(self.score, 4),
            "vector_score": _r(self.vector_score),
            "keyword_score": _r(self.keyword_score),
            "rerank_score": _r(self.rerank_score),
        }


@dataclass
class CandidateHit:
    """A candidate plus the evidence that put them in the results."""

    candidate: Any  # ORM Candidate row
    score: float = 0.0
    best_chunk: RetrievedChunk | None = None
    supporting_chunks: list[RetrievedChunk] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)

    @property
    def citations(self) -> list[RetrievedChunk]:
        return [self.best_chunk, *self.supporting_chunks] if self.best_chunk else []


@dataclass
class SearchOutcome:
    """Full result of a search, including diagnostics for the UI/eval."""

    hits: list[CandidateHit] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    filters_applied: dict = field(default_factory=dict)
    retrieval_ms: float = 0.0
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filters_applied": self.filters_applied,
            "retrieval_ms": round(self.retrieval_ms, 2),
            "counts": self.counts,
        }


def _r(value: float | None) -> float | None:
    return round(value, 4) if isinstance(value, (int, float)) else None


class HybridSearcher:
    """Runs the four-stage pipeline described in the module docstring."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.store = PgVectorStore(session)

    async def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        candidate_ids: list[str] | None = None,
    ) -> SearchOutcome:
        started = time.perf_counter()
        filters = filters or SearchFilters()
        top_k = top_k or settings.top_k
        # Over-fetch, then rerank, then cut. This is standard retrieval practice.
        fetch_k = max(top_k * settings.candidate_multiplier, top_k)

        # ---- 1. structured filters -------------------------------------
        allowed_ids = candidate_ids
        if not filters.is_empty():
            ids, _, _ = await resolve_filtered_candidate_ids(self.session, filters)
            allowed_ids = ids
            if not ids:
                logger.info("filters excluded every candidate")
                return SearchOutcome(
                    hits=[],
                    filters_applied=filters.to_dict(),
                    retrieval_ms=(time.perf_counter() - started) * 1000,
                    counts={"filter_matches": 0},
                )
        if allowed_ids is not None and not allowed_ids:
            return SearchOutcome(
                filters_applied=filters.to_dict(),
                retrieval_ms=(time.perf_counter() - started) * 1000,
            )

        # ---- 2a/2b. two retrieval legs ---------------------------------
        embedder = get_embedder()
        query_vector = await embedder.embed_query(query)

        # The legs are conceptually independent, but they are executed
        # SEQUENTIALLY on purpose. A SQLAlchemy AsyncSession is not safe for
        # concurrent use - it holds a single DBAPI connection, so issuing both
        # queries through `asyncio.gather` raises
        #   "This session is provisioning a new connection; concurrent
        #    operations are not permitted"
        # as soon as the pool has to open a new connection. Two indexed
        # queries cost single-digit milliseconds, so the parallelism was not
        # buying anything anyway. For real parallelism, give each leg its own
        # AsyncSession from the pool.
        vector_rows = await self.store.semantic_chunk_search(
            query_vector, limit=fetch_k, candidate_ids=allowed_ids
        )
        # The keyword leg needs alias expansion: an HR user types "postgres"
        # while the resume says "PostgreSQL". Without this the BM25 leg
        # silently returns nothing for half the queries people actually ask.
        keyword_rows = await self.store.keyword_chunk_search(
            expand_query_for_skills(query), limit=fetch_k, candidate_ids=allowed_ids
        )

        # ---- 3. reciprocal rank fusion ---------------------------------
        fused = reciprocal_rank_fusion(
            vector_rows, keyword_rows, rrf_k=settings.rrf_k
        )

        # ---- 4. optional cross-encoder rerank --------------------------
        rerank_scores = await rerank_chunks(query, [c.content for c in fused], enabled=use_rerank)
        if rerank_scores:
            scaled = normalize_scores(rerank_scores)
            for chunk, raw, scaled_value in zip(fused, rerank_scores, scaled, strict=True):
                chunk.rerank_score = raw
                # Reranking is the most reliable signal we have, so it dominates
                # the fused score while RRF keeps a tie-breaker for coverage.
                chunk.score = 0.75 * scaled_value + 0.25 * chunk.score

        fused.sort(key=lambda c: c.score, reverse=True)
        chunks = fused[:fetch_k]

        # ---- 5. collapse chunks into candidates -------------------------
        hits = await self._group_by_candidate(chunks, top_k)

        outcome = SearchOutcome(
            hits=hits,
            chunks=chunks[: top_k * 2],
            filters_applied=filters.to_dict(),
            retrieval_ms=(time.perf_counter() - started) * 1000,
            counts={
                "vector_hits": len(vector_rows),
                "keyword_hits": len(keyword_rows),
                "fused_chunks": len(fused),
                "candidates_returned": len(hits),
                "filter_matches": len(allowed_ids) if allowed_ids is not None else None,
            },
        )
        logger.info(
            "hybrid search '%s': vec=%d kw=%d fused=%d candidates=%d in %.0fms",
            query[:50], len(vector_rows), len(keyword_rows), len(fused), len(hits),
            outcome.retrieval_ms,
        )
        return outcome

    async def _group_by_candidate(
        self, chunks: list[RetrievedChunk], top_k: int
    ) -> list[CandidateHit]:
        """
        One row per candidate.

        The chunk with the highest score becomes `best_chunk` (this is what the
        LLM sees first); up to 3 more become supporting evidence. This keeps the
        prompt small while still letting the answer cite several places.
        """
        if not chunks:
            return []

        order: list[str] = []
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            cid = str(chunk.candidate_id)
            if cid not in grouped:
                grouped[cid] = []
                order.append(cid)
            grouped[cid].append(chunk)

        candidates = {
            str(c.id): c
            for c in await self.store.get_candidates(order[: top_k * 3])
        }

        hits: list[CandidateHit] = []
        for cid in order:
            candidate = candidates.get(cid)
            if candidate is None:
                continue  # deleted between search and fetch
            group = grouped[cid]
            hits.append(
                CandidateHit(
                    candidate=candidate,
                    score=group[0].score,
                    best_chunk=group[0],
                    supporting_chunks=group[1:4],
                    matched_skills=[s.skill for s in getattr(candidate, "skill_links", []) or []],
                )
            )
            if len(hits) >= top_k:
                break
        return hits


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    vector_rows: list[dict],
    keyword_rows: list[dict],
    *,
    rrf_k: int = 60,
    weight_vector: float = 1.0,
    weight_keyword: float = 1.0,
) -> list[RetrievedChunk]:
    """
    Fuse two ranked lists by rank (not by score).

    `vector_rows` / `keyword_rows` are the dicts returned by PgVectorStore,
    already ordered best-first.
    """
    merged: dict[str, RetrievedChunk] = {}
    rrf: dict[str, float] = {}

    def absorb(rows: list[dict], rank_field: str, weight: float) -> None:
        for rank, row in enumerate(rows, start=1):
            cid = str(row["id"])
            chunk = merged.get(cid)
            if chunk is None:
                chunk = RetrievedChunk(
                    chunk_id=cid,
                    candidate_id=str(row["candidate_id"]),
                    chunk_index=row["chunk_index"],
                    section=row.get("section") or "other",
                    heading=row.get("heading"),
                    content=row["content"],
                )
                merged[cid] = chunk
            # Keep the raw leg scores for explainability in the UI.
            if rank_field == "vector_rank":
                chunk.vector_rank = rank
                chunk.vector_score = row.get("vector_score")
            else:
                chunk.keyword_rank = rank
                chunk.keyword_score = row.get("keyword_score")
            # The core of RRF: only the position matters.
            rrf[cid] = rrf.get(cid, 0.0) + weight / (rrf_k + rank)

    absorb(vector_rows, "vector_rank", weight_vector)
    absorb(keyword_rows, "keyword_rank", weight_keyword)

    for cid, score in rrf.items():
        merged[cid].score = score

    return sorted(merged.values(), key=lambda c: c.score, reverse=True)