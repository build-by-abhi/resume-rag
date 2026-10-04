"""
Resume chunking.

WHY THIS IS NOT JUST "split every 1000 characters"
--------------------------------------------------
A resume has natural structure (Summary / Experience / Education / Skills).
Retrieval quality depends on chunks being *self-contained units*: if a chunk
starts mid-bullet ("...responsible for the migration to Postgres") the
embedding has no idea who/what it belongs to. So we:

1. Detect section headers (they are short, standalone, and match known keywords)
   and never merge content across a header boundary.
2. Prepend a breadcrumb ("Experience at Google") to every chunk so an isolated
   chunk still carries its context into the embedding and into the LLM prompt.
3. Chunk *within* a section on paragraph boundaries, with a token overlap so a
   fact that straddles a cut is still fully present in at least one chunk.
4. Attach a stable `heading` + `section` so results can be cited as
   "Experience | Google | 2021-2024, chunk 3".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.utils.text import approx_token_count, normalize_space

# Ordered longest-first so "professional experience" wins over "experience".
SECTION_PATTERNS: list[tuple[str, str]] = [
    ("summary", r"\b(professional\s+)?(summary|profile|about(\s+me)?|objective|overview)\b"),
    (
        "experience",
        r"\b(work\s+|professional\s+|relevant\s+|technical\s+)?"
        r"(experience|employment|work\s+history|career)\b",
    ),
    ("projects", r"\b(projects?|portfolio|selected\s+work)\b"),
    ("education", r"\b(education|academic\s+background|academics|qualifications)\b"),
    (
        "skills",
        r"\b(technical\s+skills|core\s+competencies|skills(\s*&?\s*tools)?"
        r"|technologies|tech\s+stack)\b",
    ),
    ("certifications", r"\b(certifications?|licen[cs]es?|credentials)\b"),
    ("languages", r"\b(languages?)\b"),
    ("publications", r"\b(publications?|research|papers|patents)\b"),
    ("awards", r"\b(awards?|honors?|achievements?)\b"),
    ("other", r"\b(activities|interests|hobbies|volunteering|references?)\b"),
]

# A "header" line is short, has no sentence punctuation, and looks like a title.
_HEADER_MAX_CHARS = 60
_HEADER_LOOKS_LIKE_TITLE = re.compile(r"^[A-Z][A-Za-z /&+,\-()'’.]{1,50}$")


@dataclass
class Chunk:
    """A retrievable slice of the resume, before it becomes a DB row."""

    chunk_index: int
    section: str
    content: str
    token_count: int
    heading: str | None = None
    char_start: int = 0
    char_end: int = 0
    extra: dict = field(default_factory=dict)


class ResumeChunker:
    """
    Configurable chunker.

    Args:
        chunk_size:   target size in *approximate tokens* per chunk
        chunk_overlap: how many tokens to repeat from the previous chunk
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        if self.chunk_overlap >= self.chunk_size:
            # Silently degrade rather than loop forever.
            self.chunk_overlap = max(0, self.chunk_size // 4)

    # -- public API ---------------------------------------------------------

    def chunk(self, text: str) -> list[Chunk]:
        """Split a whole resume into overlapping, section-aware chunks."""
        text = (text or "").strip()
        if not text:
            return []

        sections = self._split_sections(text)
        chunks: list[Chunk] = []
        cursor = 0

        for section_name, section_title, blocks in sections:
            for heading, body in self._split_blocks(blocks):
                for piece in self._pack(body, heading):
                    content = self._with_breadcrumb(section_title, heading, piece)
                    if not content.strip():
                        continue
                    chunks.append(
                        Chunk(
                            chunk_index=len(chunks),
                            section=section_name,
                            content=content,
                            token_count=approx_token_count(content),
                            heading=heading,
                            char_start=cursor,
                            char_end=cursor + len(piece),
                            extra={"section_title": section_title},
                        )
                    )
                    cursor += len(piece)
        return chunks

    # -- internals ---------------------------------------------------------

    def _split_sections(
        self, text: str
    ) -> list[tuple[str, str, list[str]]]:
        """
        Break the resume into (section_key, section_title, lines) groups.

        Lines before the first recognised header become the implicit "summary"
        block, which is exactly where names, emails and phones live -- so we
        make sure they are never dropped.
        """
        lines = text.split("\n")
        result: list[tuple[str, str, list[str]]] = []
        current_key = "summary"
        current_title = ""
        buffer: list[str] = []

        for line in lines:
            key = self._detect_header(line)
            if key:
                if any(b.strip() for b in buffer):
                    result.append((current_key, current_title, buffer))
                current_key = key
                # The header line itself becomes the breadcrumb prefix, so keep
                # its literal text rather than the canonical key.
                current_title = line.strip().strip(":").strip()
                buffer = []
                continue
            buffer.append(line)

        if any(b.strip() for b in buffer):
            result.append((current_key, current_title, buffer))

        # Merge away empty sections but keep order.
        return [r for r in result if any(b.strip() for b in r[2])]

    def _detect_header(self, line: str) -> str | None:
        """Return the section key if `line` looks like a section header."""
        stripped = line.strip().strip(":").strip()
        if not stripped or len(stripped) > _HEADER_MAX_CHARS:
            return None
        if not _HEADER_LOOKS_LIKE_TITLE.match(stripped):
            return None

        # Normalise: "PROFESSIONAL EXPERIENCE" -> "professional experience"
        probe = normalize_space(stripped).rstrip(":")
        probe_no_symbols = re.sub(r"[^a-z ]", "", probe).strip()

        for key, pattern in SECTION_PATTERNS:
            if re.search(pattern, probe_no_symbols, re.IGNORECASE):
                # "experience" header must not fire on "5 years of experience".
                return key
        return None

    def _split_blocks(self, lines: list[str]) -> list[tuple[str | None, str]]:
        """
        Group lines into (heading, body) units.

        A heading is a short line that does NOT end with a period and is not
        indented as a continuation - typical "Senior Engineer | Acme | 2021".
        That gives us citable chunk headings without an LLM call.
        """
        blocks: list[tuple[str | None, str]] = []
        heading: str | None = None
        body: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Blank line = hard boundary inside the current block.
                if body:
                    blocks.append((heading, "\n".join(body).strip()))
                    heading, body = None, []
                continue

            if self._looks_like_entry_heading(stripped):
                if body:
                    blocks.append((heading, "\n".join(body).strip()))
                heading = stripped
                body = [stripped]
                continue

            body.append(stripped)

        if body:
            blocks.append((heading, "\n".join(body).strip()))

        return [(h, b) for h, b in blocks if b.strip()]

    @staticmethod
    def _looks_like_entry_heading(line: str) -> bool:
        if len(line) > 80 or line.endswith((".", ",", ";", ":")):
            return False
        words = line.split()
        if not (1 <= len(words) <= 12):
            return False
        # Mostly title-case words => likely a role/company/degree line.
        titleish = sum(1 for w in words if w[:1].isupper() or w.isupper())
        return titleish >= max(1, len(words) - 2)

    def _pack(self, body: str, heading: str | None) -> list[str]:
        """
        Split one block into <= chunk_size pieces with token overlap.

        Prefers paragraph breaks, then sentence boundaries, then hard cuts --
        so a chunk almost never ends mid-sentence.
        """
        if approx_token_count(body) <= self.chunk_size:
            return [body]

        units = self._split_into_units(body)
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0
        # Hard ceiling: a unit that survives splitting can still be over budget.
        # Without this guard a pathological unit could produce zero-length
        # pieces and spin forever.
        hard_cap = max(1, self.chunk_size)

        for unit in units:
            unit_tokens = approx_token_count(unit)

            # Single unit bigger than the budget (very long paragraph):
            # emit it on its own, then hard-split it.
            if unit_tokens > self.chunk_size:
                if current:
                    pieces.append("\n\n".join(current))
                    current, current_tokens = [], 0
                pieces.extend(self._hard_split(unit))
                continue

            over_budget = current_tokens + unit_tokens > self.chunk_size
            if current and (over_budget or current_tokens > hard_cap):
                pieces.append("\n\n".join(current))
                current = self._tail_for_overlap(current)
                current_tokens = sum(approx_token_count(c) for c in current)

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            pieces.append("\n\n".join(current))
        return pieces

    def _split_into_units(self, body: str) -> list[str]:
        """Paragraphs -> sentences -> words, whichever granularity fits."""
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        units: list[str] = []
        for para in paragraphs:
            if approx_token_count(para) <= self.chunk_size:
                units.append(para)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buffer: list[str] = []
            buffer_tokens = 0
            for sentence in sentences:
                st = approx_token_count(sentence)
                if buffer and buffer_tokens + st > self.chunk_size:
                    units.append(" ".join(buffer))
                    buffer, buffer_tokens = [], 0
                buffer.append(sentence)
                buffer_tokens += st
            if buffer:
                units.append(" ".join(buffer))
        return units

    def _tail_for_overlap(self, units: list[str]) -> list[str]:
        """Last units that fit in the overlap budget, so context carries over."""
        if self.chunk_overlap <= 0:
            return []
        tail: list[str] = []
        tokens = 0
        for unit in reversed(units):
            ut = approx_token_count(unit)
            if tokens + ut > self.chunk_overlap:
                break
            tail.insert(0, unit)
            tokens += ut
        return tail

    def _hard_split(self, text: str) -> list[str]:
        """Last-resort split for pathological input (no word/paragraph breaks)."""
        window_chars = max(1, self.chunk_size * 4)
        step_chars = max(1, window_chars - self.chunk_overlap * 4)

        words = text.split()
        # A word-window split only works when there are many words. A single
        # 20k-character "word" (base64 blob, minified JSON, broken OCR) has to
        # be cut by character position instead.
        if len(words) > window_chars // 4:
            step_words = max(1, step_chars // 4)
            return [
                " ".join(words[i : i + window_chars // 4])
                for i in range(0, len(words), step_words)
            ]

        return [text[i : i + window_chars] for i in range(0, len(text), step_chars)]

    @staticmethod
    def _with_breadcrumb(section_title: str, heading: str | None, body: str) -> str:
        """
        Prepend context to the chunk text.

        This matters a lot: the embedding of "Designed a Kafka pipeline..." is
        much more useful when it also says "EXPERIENCE" and "Acme Corp".
        """
        prefix_bits: list[str] = []
        if section_title:
            prefix_bits.append(section_title)
        # Entry headings are already the first line of the body -> don't repeat.
        if heading and not body.startswith(heading):
            prefix_bits.append(heading)
        prefix = f"[{' | '.join(prefix_bits)}]\n" if prefix_bits else ""
        return f"{prefix}{body}"


# Shared instance - chunking is stateless, so one is enough.
chunker = ResumeChunker()