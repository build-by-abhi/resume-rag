"""Candidate-facing schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class SkillOut(ORMModel):
    canonical: str
    display: str
    occurrences: int = 1
    is_core: bool = False


class EducationOut(ORMModel):
    level: str | None = None
    field: str | None = None
    institution: str | None = None
    gpa: float | None = None


class CandidateSummary(ORMModel):
    """Compact row used in lists and search results."""

    id: UUID
    full_name: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    location: str | None = None
    total_years_experience: float | None = None
    seniority: str | None = None
    email: str | None = None
    phone: str | None = None
    summary: str | None = Field(None, max_length=1200)
    status: str = "ready"
    created_at: datetime | None = None


class ChunkOut(ORMModel):
    id: UUID
    chunk_index: int
    section: str
    heading: str | None = None
    content: str
    token_count: int = 0


class CandidateDetail(CandidateSummary):
    """Full record: used by the candidate detail page."""

    links: dict[str, Any] = Field(default_factory=dict)
    education: dict[str, Any] = Field(default_factory=dict)
    languages: list[str] = Field(default_factory=list)
    skills: list[SkillOut] = Field(default_factory=list)
    resume_filename: str | None = None
    extraction_method: str | None = None
    text_length: int = 0
    chunk_count: int = 0
    chunks: list[ChunkOut] = Field(default_factory=list)
    updated_at: datetime | None = None


class IngestSummary(ORMModel):
    """Result of ingesting one file."""

    candidate_id: UUID
    created: bool
    chunk_count: int
    skill_count: int
    text_length: int
    full_name: str | None = None
    extraction_method: str | None = None
    warnings: list[str] = Field(default_factory=list)


class IngestTextRequest(BaseModel):
    """
    Body for the paste-a-resume endpoint.

    A Pydantic model (rather than a bare `text: str` parameter) is what makes
    FastAPI parse `{"text": "..."}` as JSON. A bare scalar parameter would be
    read as the entire request body.
    """

    text: str = Field(
        min_length=80,
        max_length=200_000,
        description="Full resume text",
    )


class UploadResponse(ORMModel):
    """Response for a multi-file upload (one item per file)."""

    batch_size: int
    succeeded: int
    failed: int
    results: list[IngestSummary] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)