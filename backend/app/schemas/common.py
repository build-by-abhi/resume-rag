"""Shared response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    """
    Base for schemas built from ORM rows.

    `from_attributes=True` is what lets us do `CandidateOut.model_validate(row)`
    instead of hand-mapping every field.
    """

    model_config = ConfigDict(from_attributes=True)


class LLMStatus(BaseModel):
    """Why the LLM is (or is not) available - surfaced verbatim in the UI."""

    configured: bool
    provider: str
    model: str
    reason: str | None = None
    env_var: str | None = Field(
        default=None, description="Which env var is missing, when that is the cause"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str
    database: str = Field(description="'ok', 'degraded' or 'down'")
    embedding_provider: str
    embedding_dim: int
    llm_enabled: bool
    llm_status: LLMStatus = Field(
        default_factory=lambda: LLMStatus(
            configured=False, provider="none", model="", reason="not checked"
        )
    )
    candidate_count: int = 0
    chunk_count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
    hint: str | None = None