"""Search request/response schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.candidate import SkillOut


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    """
    One endpoint covers both Phase 2 (natural language) and Phase 3 (hybrid).

    You can send:
      * a free-text question only              -> NL search
      * filters only, no question              -> pure structured search
      * a question AND filters                 -> the intended hybrid case
    """

    query: str = Field(
        default="",
        max_length=1000,
        description="Natural-language question, e.g. 'senior Python with Kubernetes'",
    )

    # -- structured filters (the exact-match half of hybrid search) -------
    skills: list[str] = Field(default_factory=list, max_length=30)
    skills_mode: str = Field(default="all", pattern="^(all|any)$")
    location: str | None = Field(default=None, max_length=160)
    min_years_experience: float | None = Field(default=None, ge=0, le=60)
    max_years_experience: float | None = Field(default=None, ge=0, le=60)
    current_title: str | None = Field(default=None, max_length=200)
    company: str | None = Field(default=None, max_length=200)
    seniority: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    education_level: list[str] = Field(default_factory=list)
    has_email: bool | None = None
    has_phone: bool | None = None

    # -- retrieval controls ----------------------------------------------
    top_k: int = Field(default=10, ge=1, le=50)
    use_rerank: bool | None = Field(
        default=None,
        description="null = follow the RERANK_ENABLED setting",
    )
    # Set false to skip the natural-language answer and return only ranked rows.
    generate_answer: bool = True

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        return (value or "").strip()

    @property
    def has_text_or_filters(self) -> bool:
        """
        Whether this request carries anything to search with.

        Mirrored by `isSearchable()` in `frontend/src/components/FilterPanel.tsx`,
        so the UI can avoid sending a request that would be rejected with 422.
        Keep the two lists in sync.
        """
        return bool(self.query) or any(
            [
                self.skills,
                self.location,
                self.min_years_experience is not None,
                self.max_years_experience is not None,
                self.current_title,
                self.company,
                self.seniority,
                self.languages,
                self.education_level,
                self.has_email is not None,
                self.has_phone is not None,
            ]
        )


class JDMatchRequest(BaseModel):
    """Phase 4: rank the whole pool against a pasted job description."""

    job_description: str = Field(min_length=40, max_length=20_000)
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0, le=100)
    generate_analysis: bool = True


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class EvidenceOut(BaseModel):
    """One retrieved chunk, exposed so the UI can show *why* a candidate matched."""

    chunk_id: str
    candidate_id: str
    chunk_index: int
    section: str
    heading: str | None = None
    content: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None


class CandidateResult(BaseModel):
    id: UUID
    full_name: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    location: str | None = None
    total_years_experience: float | None = None
    seniority: str | None = None
    summary: str | None = None
    email: str | None = None
    phone: str | None = None
    score: float
    skills: list[SkillOut] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)


class CitationOut(BaseModel):
    index: int
    candidate_id: str
    candidate_name: str | None = None
    section: str | None = None
    heading: str | None = None
    excerpt: str


class AnswerOut(BaseModel):
    text: str
    citations: list[CitationOut] = Field(default_factory=list)
    used_llm: bool = False
    model: str | None = None
    latency_ms: float | None = None
    insufficient_evidence: bool = False
    had_invalid_citations: bool = False


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[CandidateResult] = Field(default_factory=list)
    answer: AnswerOut | None = None
    filters_applied: dict = Field(default_factory=dict)
    retrieval_ms: float
    total_ms: float
    diagnostics: dict = Field(default_factory=dict)


class JDMatchResult(BaseModel):
    candidate_id: UUID
    full_name: str | None = None
    current_title: str | None = None
    score: float
    skills_match: float | None = None
    experience_match: float | None = None
    seniority_match: float | None = None
    missing_critical_skills: list[str] = Field(default_factory=list)
    strengths: list[dict] = Field(default_factory=list)
    summary: str | None = None
    evidence: list[EvidenceOut] = Field(default_factory=list)


class JDMatchResponse(BaseModel):
    job_description_excerpt: str
    total: int
    results: list[JDMatchResult] = Field(default_factory=list)
    used_llm: bool = False
    retrieval_ms: float
    total_ms: float