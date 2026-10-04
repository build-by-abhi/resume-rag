"""
The database schema. This is where the "RAG + structured" idea becomes concrete.

Three different kinds of data live here, each stored the way it is best queried:

1. `candidates`      -> structured columns (name, email, years_of_experience...)
                         Best for EXACT filtering: `location = 'Berlin'`,
                         `total_years_experience >= 5`, `skills && ARRAY['python']`.
                         Row comparison + GIN indexes are exact and instant.

2. `candidate_skills`-> one row per (candidate, skill) pair.
                         Best for "how many people know X", skill facets, and
                         per-skill aggregation.

3. `resume_chunks`   -> the actual RAG chunks: text + embedding + tsvector.
                         Best for SEMANTIC and keyword search inside prose
                         ("built a CI pipeline", "led a migration to Postgres").

Search (Phase 3) queries 1 + 2 + 3 together: structured filters + BM25 keyword
rank + vector similarity, fused with Reciprocal Rank Fusion. That is why we do
NOT use "pure vector search" for this problem.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    literal_column,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base

# The vector column width is fixed at table-creation time. If you change
# EMBEDDING_DIM in .env you must drop/recreate the tables (Phase 5 covers this
# with a proper migration).
EMBEDDING_DIM = settings.embedding_dim


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, default=uuid.uuid4)


class Candidate(Base):
    """One row per resume/candidate. The structured "profile" record."""

    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # ---- Identity -------------------------------------------------------
    full_name: Mapped[str | None] = mapped_column(String(255), index=True)
    email: Mapped[str | None] = mapped_column(String(320), index=True)
    phone: Mapped[str | None] = mapped_column(String(64))
    location: Mapped[str | None] = mapped_column(String(160), index=True)
    links: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict)

    # ---- Vitals used for filters & ranking boosts ------------------------
    current_title: Mapped[str | None] = mapped_column(String(200), index=True)
    current_company: Mapped[str | None] = mapped_column(String(200))
    total_years_experience: Mapped[float | None] = mapped_column(Float, index=True)
    seniority: Mapped[str | None] = mapped_column(String(40), index=True)
    education: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict)
    languages: Mapped[list[str]] = mapped_column(ARRAY(String(64)), default=list)

    # ---- Content --------------------------------------------------------
    summary: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text, default="")

    # A single "whole resume" vector. Used for cheap candidate-level
    # similarity and as one leg of the hybrid search.
    profile_embedding: Mapped[Any | None] = mapped_column(Vector(EMBEDDING_DIM))

    # ---- Provenance / lifecycle ----------------------------------------
    resume_filename: Mapped[str | None] = mapped_column(String(512))
    resume_path: Mapped[str | None] = mapped_column(String(1024))
    # SHA-256 of the file bytes: lets us detect re-uploads of the same resume
    # and makes ingestion idempotent (Phase 5 dedupe).
    source_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(
        String(20), default="ready", index=True
    )  # pending | processing | ready | failed
    extraction_method: Mapped[str | None] = mapped_column(String(32))  # rules | llm | hybrid
    error_message: Mapped[str | None] = mapped_column(Text)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ---- Relationships --------------------------------------------------
    chunks: Mapped[list[ResumeChunk]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", lazy="selectin"
    )
    skill_links: Mapped[list[CandidateSkill]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint("status IN ('pending','processing','ready','failed')",
                        name="ck_candidates_status"),
        # Full-text index over the structured fields, so even a short query
        # ("data engineer Berlin") can be answered by BM25 without touching chunks.
        #
        # Two Postgres details at work here:
        #  1. Index expressions must be IMMUTABLE, so we join with `||`
        #     (immutable) rather than concat_ws() (stable).
        #  2. The text-search config must be a literal, not a bind parameter --
        #     Postgres cannot store a parameterised expression in an index.
        #     `literal_column` renders `'english'` inline, which is what we want.
        Index(
            "ix_candidates_profile_tsv",
            func.to_tsvector(
                literal_column("'english'"),
                func.coalesce(full_name, "")
                + " " + func.coalesce(current_title, "")
                + " " + func.coalesce(current_company, "")
                + " " + func.coalesce(summary, "")
                + " " + func.coalesce(location, ""),
            ),
            postgresql_using="gin",
        ),
    )


class CandidateSkill(Base):
    """
    One row per (candidate, normalized skill). Normalized means "React.js",
    "reactjs", "React JS" all collapse to "react" so filters behave.
    """

    __tablename__ = "candidate_skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    # Canonical lowercase name, e.g. "python".
    skill: Mapped[str] = mapped_column(String(80), index=True)
    # Display form as written in the resume, e.g. "Python", "PyTorch".
    display: Mapped[str] = mapped_column(String(120))
    # How many times it appears -> a weak relevance signal (core skill).
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    years: Mapped[float | None] = mapped_column(Float)
    is_core: Mapped[bool] = mapped_column(default=False)

    candidate: Mapped[Candidate] = relationship(back_populates="skill_links")

    __table_args__ = (
        Index("ix_candidate_skills_unique", "candidate_id", "skill", unique=True),
        Index("ix_candidate_skills_skill_gin", "skill", postgresql_using="btree"),
    )


class ResumeChunk(Base):
    """A slice of resume text plus everything needed to search it."""

    __tablename__ = "resume_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )

    chunk_index: Mapped[int] = mapped_column(Integer)
    # 'summary' | 'experience' | 'education' | 'skills' | 'projects' | 'other'
    section: Mapped[str] = mapped_column(String(40), default="other", index=True)
    # e.g. "Google | Senior Engineer | 2021-2024" -> used for display/citations.
    heading: Mapped[str | None] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)

    # The semantic leg of hybrid search.
    embedding: Mapped[Any | None] = mapped_column(Vector(EMBEDDING_DIM))
    # The keyword leg of hybrid search: a tsvector column Postgres can rank with
    # ts_rank_cd(). It is a GENERATED column, so Postgres keeps it in sync with
    # `content` automatically -- Python never writes to it.
    search_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(content, ''))", persisted=True),
        nullable=True,
    )

    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    candidate: Mapped[Candidate] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("uq_resume_chunk", "candidate_id", "chunk_index", unique=True),
        # BM25 keyword index.
        Index("ix_resume_chunks_tsv", "search_tsv", postgresql_using="gin"),
        # Approximate nearest-neighbour index for the vector leg.
        # cosine distance matches how we normalize+compare embeddings.
        Index(
            "ix_resume_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class SearchQuery(Base):
    """
    Audit log of every HR search. Free analytics for the portfolio write-up:
    zero-result rate, popular filters, latency. Phase 5 turns this into a real
    evaluation set.
    """

    __tablename__ = "search_queries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    query_text: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    top_candidate_ids: Mapped[list[str]] = mapped_column(ARRAY(String(36)), default=list)
    retrieval_ms: Mapped[float | None] = mapped_column(Float)
    llm_ms: Mapped[float | None] = mapped_column(Float)
    used_llm: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_search_queries_created", "created_at"),)