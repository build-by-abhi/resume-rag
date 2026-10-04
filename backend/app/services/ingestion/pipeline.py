"""
The ingestion pipeline: file bytes -> searchable candidate.

This is the single place that knows the whole order of operations. Read it once
and the architecture of the project makes sense.

    ┌─ 1. parse        PDF/DOCX/TXT  ->  raw text
    ├─ 2. clean        raw           ->  normalised text (whitespace, ligatures)
    ├─ 3. dedupe       sha256 of the file; skip if already ingested
    ├─ 4. extract      text -> structured fields (rules, then optional LLM)
    ├─ 5. chunk        text -> section-aware overlapping chunks
    ├─ 6. embed        chunks -> vectors;  full text -> one profile vector
    └─ 7. persist      ONE transaction: candidate + skills + chunks

Why one transaction at the end: a partially ingested candidate (profile saved,
no chunks) is invisible to search but breaks every "expected N results" test.
All-or-nothing is the right call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Candidate, CandidateSkill, ResumeChunk
from app.services.chunking import ResumeChunker
from app.services.embeddings import get_embedder
from app.services.extraction import Extraction, merge_extractions, rule_extractor
from app.services.extraction.llm_extractor import llm_extractor
from app.services.ingestion.pdf_loader import parse_file
from app.utils.files import sha256_file
from app.utils.text import approx_token_count, clean_text

logger = get_logger(__name__)


@dataclass
class IngestResult:
    candidate: Candidate
    created: bool
    chunk_count: int
    skill_count: int
    text_chars: int


class IngestionPipeline:
    """Orchestrates parse -> clean -> extract -> chunk -> embed -> persist."""

    def __init__(
        self,
        *,
        use_llm_extraction: bool | None = None,
        chunker: ResumeChunker | None = None,
    ) -> None:
        self.chunker = chunker or ResumeChunker()
        # `None` = decide from env (LLM enabled AND a key is present).
        self.use_llm_extraction = use_llm_extraction

    async def _should_use_llm(self) -> bool:
        if self.use_llm_extraction is not None:
            return self.use_llm_extraction
        return llm_extractor.enabled

    # -- public API ---------------------------------------------------------

    async def ingest_file(
        self,
        session: AsyncSession,
        path: Path,
        *,
        original_filename: str | None = None,
        force: bool = False,
    ) -> IngestResult:
        """Ingest one already-saved file. Returns the (possibly existing) candidate."""
        path = Path(path)
        content_hash = sha256_file(path)

        # --- 3. dedupe -----------------------------------------------------
        existing = await self._find_by_hash(session, content_hash)
        if existing and not force:
            logger.info("resume already ingested (hash match): %s", content_hash[:8])
            return IngestResult(
                candidate=existing,
                created=False,
                chunk_count=len(existing.chunks),
                skill_count=len(existing.skill_links),
                text_chars=len(existing.raw_text),
            )

        # --- 1 + 2. parse and clean ---------------------------------------
        raw = parse_file(path)
        text = clean_text(raw)
        if len(text) < 80:
            raise ValueError(
                f"Extracted text is too short ({len(text)} chars). "
                "This is probably a scanned image with no text layer."
            )

        # --- 4. structured extraction --------------------------------------
        rules = rule_extractor.extract(text)
        llm_result: Extraction | None = None
        use_llm = await self._should_use_llm()
        if use_llm:
            llm_result = await llm_extractor.extract(text)
        extraction = merge_extractions(rules, llm_result, llm_enabled=use_llm)
        extraction_method = "hybrid" if use_llm else "rules"

        # --- 5. chunk ------------------------------------------------------
        chunks = self.chunker.chunk(text)

        # --- 6. embed ------------------------------------------------------
        embedder = get_embedder()
        chunk_vectors: list[list[float]] = []
        if chunks:
            chunk_vectors = await embedder.embed_documents([c.content for c in chunks])
        profile_vector = await embedder.embed_query(self._profile_summary(text, extraction))

        # --- 7. persist (single transaction) -------------------------------
        if existing:
            await self._replace_payload(
                session, existing, extraction, chunks, chunk_vectors,
                profile_vector, text, path, original_filename,
            )
            candidate = existing
            created = False
        else:
            candidate = Candidate(
                full_name=extraction.full_name,
                email=extraction.email,
                phone=extraction.phone,
                location=extraction.location,
                links=extraction.links,
                current_title=extraction.current_title,
                current_company=extraction.current_company,
                total_years_experience=extraction.total_years_experience,
                seniority=extraction.seniority,
                education=extraction.education,
                languages=extraction.languages,
                summary=extraction.summary,
                raw_text=text,
                profile_embedding=profile_vector,
                resume_filename=original_filename or path.name,
                resume_path=str(path),
                source_hash=content_hash,
                status="ready",
                extraction_method=extraction_method,
            )
            candidate.chunks = self._build_chunks(chunks, chunk_vectors, candidate.id)
            candidate.skill_links = self._build_skills(extraction, candidate.id)
            session.add(candidate)
            created = True

        await session.commit()
        await session.refresh(candidate)

        logger.info(
            "ingested %s: candidate=%s chunks=%d skills=%d method=%s",
            path.name,
            candidate.id,
            len(chunks),
            len(extraction.skills),
            extraction_method,
        )
        return IngestResult(
            candidate=candidate,
            created=created,
            chunk_count=len(chunks),
            skill_count=len(extraction.skills),
            text_chars=len(text),
        )

    # -- helpers ------------------------------------------------------------

    async def _find_by_hash(self, session: AsyncSession, content_hash: str) -> Candidate | None:
        result = await session.execute(
            select(Candidate).where(Candidate.source_hash == content_hash)
        )
        return result.scalar_one_or_none()

    def _build_chunks(self, chunks, vectors, candidate_id) -> list[ResumeChunk]:
        rows: list[ResumeChunk] = []
        # strict=True turns a length mismatch into a loud error instead of
        # silently dropping chunks.
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows.append(
                ResumeChunk(
                    candidate_id=candidate_id,
                    chunk_index=chunk.chunk_index,
                    section=chunk.section,
                    heading=chunk.heading,
                    content=chunk.content,
                    token_count=chunk.token_count or approx_token_count(chunk.content),
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    embedding=vector,
                    extra=chunk.extra,
                )
            )
        return rows

    @staticmethod
    def _build_skills(extraction: Extraction, candidate_id) -> list[CandidateSkill]:
        rows: list[CandidateSkill] = []
        for canonical, meta in (extraction.skills or {}).items():
            rows.append(
                CandidateSkill(
                    candidate_id=candidate_id,
                    skill=canonical,
                    display=meta.get("display", canonical),
                    occurrences=int(meta.get("occurrences", 1)),
                    is_core=bool(meta.get("is_core", False)),
                )
            )
        return rows

    async def _replace_payload(
        self,
        session: AsyncSession,
        candidate: Candidate,
        extraction: Extraction,
        chunks,
        vectors,
        profile_vector,
        text: str,
        path: Path,
        original_filename: str | None,
    ) -> None:
        """Re-ingesting the same file: swap the payload, keep the same row."""
        candidate.full_name = extraction.full_name
        candidate.email = extraction.email
        candidate.phone = extraction.phone
        candidate.location = extraction.location
        candidate.links = extraction.links
        candidate.current_title = extraction.current_title
        candidate.current_company = extraction.current_company
        candidate.total_years_experience = extraction.total_years_experience
        candidate.seniority = extraction.seniority
        candidate.education = extraction.education
        candidate.languages = extraction.languages
        candidate.summary = extraction.summary
        candidate.raw_text = text
        candidate.profile_embedding = profile_vector
        candidate.resume_filename = original_filename or path.name
        candidate.resume_path = str(path)
        candidate.status = "ready"
        candidate.error_message = None

        # Orphan-delete: the relationship cascade removes chunks and skills,
        # then we insert the fresh ones.
        candidate.chunks = self._build_chunks(chunks, vectors, candidate.id)
        candidate.skill_links = self._build_skills(extraction, candidate.id)
        session.add(candidate)
        await session.flush()

    @staticmethod
    def _profile_summary(text: str, extraction: Extraction) -> str:
        """
        The text that becomes the single profile-level vector.

        The whole resume in one vector is blurry (it averages every topic
        together). A compact "identity card" gives a much cleaner candidate-level
        similarity signal, while the chunks handle the detail.
        """
        head = " ".join(text[:1200].split())
        parts = [
            extraction.current_title or "",
            extraction.current_company or "",
            extraction.location or "",
            f"{extraction.total_years_experience:g} years experience"
            if extraction.total_years_experience
            else "",
            " ".join(meta["display"] for meta in list(extraction.skills.values())[:20]),
            head,
        ]
        return " | ".join(p for p in parts if p)


pipeline = IngestionPipeline()