"""
Declared output shapes for the MCP tools.

WHY THIS FILE EXISTS
--------------------
The SDK derives a tool's `outputSchema` from the handler's return annotation.
Annotating `-> dict` produces NO schema, which means the client gets prose it
has to parse and the model has no field names to reference.

`TypedDict` generates a real JSON Schema:

    -> dict                     no outputSchema
    -> dict[str, Any]           empty outputSchema  (worse - looks valid)
    -> SearchResult             correct outputSchema

Each type uses `total=False` because every handler can also return an `error`
key instead of its normal payload - a tool that fails should say so in a shape
the model can branch on, not raise into the conversation.

WHY TYPEDDICT AND NOT A PYDANTIC MODEL
--------------------------------------
TypedDict describes the wire format without forcing validation on the way in or
coupling the schema to a runtime class. The data comes straight out of
`group_hits_for_display`, which is already dict-shaped.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Evidence(TypedDict, total=False):
    """One retrieved resume chunk. This is what makes a claim checkable."""

    section: str
    heading: str | None
    content: str


class SearchResultRow(TypedDict, total=False):
    id: str
    full_name: str | None
    current_title: str | None
    current_company: str | None
    location: str | None
    total_years_experience: float | None
    seniority: str | None
    score: float
    skills: list[str]
    evidence: list[Evidence]


class SearchResult(TypedDict, total=False):
    total: int
    results: list[SearchResultRow]
    diagnostics: dict[str, Any]
    # Present when the LLM wrote a summary instead of us returning rows only.
    answer: str
    answer_uses_llm: bool
    answer_warning: str
    # Present instead of results when the request itself was unusable.
    error: str


class MatchRow(TypedDict, total=False):
    id: str
    full_name: str | None
    current_title: str | None
    current_company: str | None
    location: str | None
    total_years_experience: float | None
    seniority: str | None
    retrieval_score: float
    evidence: list[dict[str, Any]]


class MatchResult(TypedDict, total=False):
    total: int
    scoring: str
    note: str
    results: list[MatchRow]
    error: str


class SkillRow(TypedDict, total=False):
    canonical: str
    display: str
    candidate_count: int
    is_core_skill: bool


class SkillResult(TypedDict, total=False):
    total: int
    skills: list[SkillRow]
    error: str


class IngestResult(TypedDict, total=False):
    candidate_id: str
    created: bool
    deduplicated: bool
    full_name: str | None
    chunk_count: int
    skill_count: int
    extraction_method: str | None
    warnings: list[str]
    error: str


__all__ = [
    "Evidence",
    "SearchResult",
    "SearchResultRow",
    "MatchResult",
    "MatchRow",
    "SkillResult",
    "SkillRow",
    "IngestResult",
]