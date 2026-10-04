"""
Turn retrieved evidence into an answer (Phase 2).

Two paths, same interface:

1. **LLM path** - send the numbered context blocks to a model and get a cited
   answer. Higher quality, costs money and latency.
2. **Extractive path** - no API key configured? Build a factual digest straight
   from the top chunks. No hallucination risk, zero cost, still useful.

The extractive path is not a placeholder for the "real" answer; it is the
guaranteed-correct floor. A recruiter asking "do we have anyone with Kubernetes
experience?" should never get a blank box because a key expired.

CITATION HYGIENE
----------------
We number the excerpts and then verify the citations the model produced. A
citation to `[9]` when only 4 excerpts exist is a hallucination, and we strip
it and flag the answer instead of passing a broken reference to the UI.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.prompts.templates import build_search_prompt
from app.services.llm.client import LLMResponse, NullLLMClient, get_llm_client
from app.services.retrieval.hybrid import CandidateHit, SearchOutcome
from app.utils.text import truncate

logger = get_logger(__name__)

# Matches [1], [2,3], [4][5]
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
MAX_CONTEXT_CHUNKS = 14  # keeps the prompt small and the latency low


@dataclass
class Citation:
    """A single evidence reference shown in the UI under the answer."""

    index: int
    candidate_id: str
    candidate_name: str | None
    section: str | None
    heading: str | None
    excerpt: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    used_llm: bool = False
    model: str | None = None
    latency_ms: float | None = None
    #: True when the model cited references that did not exist and we removed them.
    had_invalid_citations: bool = False
    insufficient_evidence: bool = False


class AnswerGenerator:
    """Builds the context block, calls the LLM, and validates the result."""

    def __init__(self, client=None) -> None:
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    async def answer(
        self,
        question: str,
        outcome: SearchOutcome,
        *,
        max_hits: int = 5,
    ) -> Answer:
        started = time.perf_counter()
        blocks, citations = self._build_context(outcome, max_hits=max_hits)

        if not blocks:
            return Answer(
                text=(
                    "No candidates in the database matched this query or the "
                    "selected filters. Try broadening the filters or uploading "
                    "more resumes."
                ),
                used_llm=False,
                insufficient_evidence=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if isinstance(self.client, NullLLMClient):
            logger.info("no LLM configured -> returning extractive summary")
            return Answer(
                text=self._extractive_answer(question, outcome),
                citations=citations,
                used_llm=False,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        system, user = build_search_prompt(question, blocks)
        try:
            response: LLMResponse = await self.client.complete(system, user, max_tokens=700)
        except Exception as exc:
            logger.warning("answer generation failed, falling back: %s", exc)
            return Answer(
                text=self._extractive_answer(question, outcome),
                citations=citations,
                used_llm=False,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        text, invalid = self._sanitize_citations(response.text, len(blocks))

        if not text.strip():
            # Never hand the UI an empty answer panel. A provider can return a
            # blank response (truncation, safety filter, provider hiccup) and an
            # empty box reads as "we found nothing", which is a different and
            # wrong conclusion. Fall back to the extractive digest.
            logger.warning(
                "LLM returned an empty answer (finish_reason=%s); "
                "serving the extractive summary instead",
                response.finish_reason,
            )
            return Answer(
                text=self._extractive_answer(question, outcome),
                citations=citations,
                used_llm=False,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return Answer(
            text=text,
            citations=citations,
            used_llm=True,
            model=response.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            had_invalid_citations=invalid,
        )

    # -- context ------------------------------------------------------------

    def _build_context(
        self, outcome: SearchOutcome, *, max_hits: int
    ) -> tuple[list[tuple[str, str]], list[Citation]]:
        """
        Flatten the hits into numbered excerpts, at most two chunks per candidate.

        Two per candidate keeps one strong candidate from crowding everyone else
        out of the context window.
        """
        blocks: list[tuple[str, str]] = []
        citations: list[Citation] = []

        for hit in outcome.hits[:max_hits]:
            for chunk in hit.citations[:2]:
                index = len(blocks) + 1
                if index > MAX_CONTEXT_CHUNKS:
                    return blocks, citations
                name = hit.candidate.full_name or "Unknown candidate"
                label = f"{name}"
                if hit.candidate.current_title:
                    label += f" ({hit.candidate.current_title})"
                blocks.append((str(index), f"{label}\n{chunk.content}"))
                citations.append(
                    Citation(
                        index=index,
                        candidate_id=str(hit.candidate.id),
                        candidate_name=hit.candidate.full_name,
                        section=chunk.section,
                        heading=chunk.heading,
                        excerpt=truncate(chunk.content, 300),
                    )
                )
        return blocks, citations

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _sanitize_citations(text: str, valid_count: int) -> tuple[str, bool]:
        """Remove citation markers that point at non-existent excerpts."""
        if not text:
            return "", False
        invalid = False

        def replace(match: re.Match[str]) -> str:
            nonlocal invalid
            kept = []
            for part in match.group(1).split(","):
                try:
                    idx = int(part.strip())
                except ValueError:
                    invalid = True
                    return ""
                if 1 <= idx <= valid_count:
                    kept.append(str(idx))
                else:
                    invalid = True
            return f"[{','.join(kept)}]" if kept else ""

        return _CITATION.sub(replace, text).strip(), invalid

    # -- extractive fallback -------------------------------------------------

    @staticmethod
    def _extractive_answer(question: str, outcome: SearchOutcome) -> str:
        """
        Deterministic digest of the top hits.

        Clearly a fallback, but it contains no generated claims - only text we
        literally read out of the resumes.
        """
        if not outcome.hits:
            return "No matching candidates found."

        lines = [
            f"Found {len(outcome.hits)} matching candidate"
            f"{'s' if len(outcome.hits) != 1 else ''} for: '{question}'",
            "",
        ]
        for hit in outcome.hits[:5]:
            candidate = hit.candidate
            name = candidate.full_name or "Unknown candidate"
            details = [
                d
                for d in (
                    candidate.current_title,
                    candidate.current_company,
                    candidate.location,
                    f"{candidate.total_years_experience:g} yrs experience"
                    if candidate.total_years_experience
                    else None,
                )
                if d
            ]
            skill_links = (getattr(candidate, "skill_links", []) or [])[:10]
            skills = ", ".join(s.display for s in skill_links)
            lines.append(f"- {name} — {' | '.join(details)}")
            if skills:
                lines.append(f"  Skills: {skills}")
            if hit.best_chunk:
                lines.append(f"  Match: {truncate(hit.best_chunk.content, 200)}")
        lines.append("")
        lines.append(
            "_LLM is not configured, so this is an extractive summary of the top "
            "matches. Set LLM_PROVIDER and the matching API key for a written answer._"
        )
        return "\n".join(lines)


answer_generator = AnswerGenerator()


def group_hits_for_display(hits: list[CandidateHit]) -> list[dict]:
    """Flatten CandidateHit objects into the exact JSON the frontend renders."""
    out: list[dict] = []
    for hit in hits:
        candidate = hit.candidate
        skill_links = sorted(
            getattr(candidate, "skill_links", []) or [],
            key=lambda s: (not s.is_core, -s.occurrences),
        )
        out.append(
            {
                "id": str(candidate.id),
                "full_name": candidate.full_name,
                "current_title": candidate.current_title,
                "current_company": candidate.current_company,
                "location": candidate.location,
                "total_years_experience": candidate.total_years_experience,
                "seniority": candidate.seniority,
                "summary": candidate.summary,
                "score": round(hit.score, 4),
                "skills": [
                    {
                        "canonical": s.skill,
                        # Field names must match `SkillOut` (display / is_core).
                        "display": s.display,
                        "occurrences": s.occurrences,
                        "is_core": s.is_core,
                    }
                    for s in skill_links
                ],
                "evidence": [c.to_dict() for c in hit.citations],
            }
        )
    return out