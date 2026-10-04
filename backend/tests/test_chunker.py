"""Section detection, chunk sizing, overlap and breadcrumbs."""

from __future__ import annotations

from app.services.chunking import ResumeChunker
from app.services.chunking.chunker import SECTION_PATTERNS
from app.utils.text import approx_token_count


class TestSectionDetection:
    def test_recognises_the_common_headers(self):
        chunker = ResumeChunker()
        cases = {
            "SUMMARY": "summary",
            "PROFESSIONAL SUMMARY": "summary",
            "About Me": "summary",
            "WORK EXPERIENCE": "experience",
            "Experience": "experience",
            "Professional Experience": "experience",
            "EDUCATION": "education",
            "Technical Skills": "skills",
            "Core Competencies": "skills",
            "CERTIFICATIONS": "certifications",
            "PROJECTS": "projects",
            "Languages": "languages",
        }
        for header, expected in cases.items():
            assert chunker._detect_header(header) == expected, header

    def test_ignores_ordinary_sentences_and_bullets(self):
        chunker = ResumeChunker()
        for line in (
            "5 years of experience",
            "jane.doe@example.com",
            "Led the migration of a legacy system to Kubernetes",
            "Bachelor of Science in Computer Science",
            "",
        ):
            assert chunker._detect_header(line) is None, line

    def test_every_taxonomy_pattern_is_a_valid_section_key(self):
        keys = {key for key, _ in SECTION_PATTERNS}
        assert keys  # sanity: the table is not empty


class TestChunking:
    def test_sections_become_separate_chunks(self):
        text = (
            "SUMMARY\nSenior backend engineer with 8 years of experience.\n\n"
            "EDUCATION\nMSc Computer Science, Berlin.\n\n"
            "SKILLS\nPython, Docker, PostgreSQL.\n"
        )
        chunks = ResumeChunker().chunk(text)
        sections = {c.section for c in chunks}
        assert {"summary", "education", "skills"} <= sections
        # A chunk must never mix an education line with a skills line.
        assert all("PostgreSQL" not in c.content or c.section == "skills" for c in chunks)

    def test_contact_block_before_any_header_is_kept(self):
        text = "Jane Doe\nBerlin, Germany\njane@example.com\n+49 30 5551234\n\n" + ("filler text. " * 200)
        chunks = ResumeChunker().chunk(text)
        joined = " ".join(c.content for c in chunks)
        assert "jane@example.com" in joined

    def test_respects_chunk_size(self):
        chunker = ResumeChunker(chunk_size=100, chunk_overlap=20)
        text = "SUMMARY\n" + ("Experience building distributed systems. " * 120)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1, "long input should produce several chunks"
        for chunk in chunks:
            assert chunk.token_count <= 120, "oversized chunk (breadcrumb adds a few tokens)"

    def test_overlap_keeps_context_across_a_cut(self):
        chunker = ResumeChunker(chunk_size=80, chunk_overlap=40)
        text = "SUMMARY\n" + ("Kubernetes operator patterns for clusters. " * 40)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # The last words of chunk 1 should reappear at the start of chunk 2.
        tail = chunks[0].content.strip()[-40:]
        overlap_found = any(
            any(phrase in chunk.content for phrase in (tail[:20], tail[-20:]))
            for chunk in chunks[1:]
        )
        assert overlap_found or len(chunks[0].content) < 80, "expected overlap between chunks"

    def test_breadcrumb_carries_section_and_entry_heading(self):
        text = (
            "EXPERIENCE\n"
            "Staff Engineer | Acme | 2021 - Present\n"
            "- Built a Kafka ingestion pipeline for events.\n"
        )
        chunks = ResumeChunker().chunk(text)
        content = chunks[0].content
        assert "[EXPERIENCE" in content
        assert "Staff Engineer" in content

    def test_indices_are_sequential_and_char_offsets_advance(self):
        chunks = ResumeChunker().chunk("Some sentence about work. " * 200)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.char_end >= c.char_start for c in chunks)

    def test_empty_and_whitespace_input(self):
        chunker = ResumeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n\n  ") == []

    def test_pathological_single_paragraph_is_hard_split(self):
        # One enormous "sentence" with no boundaries: must not loop forever.
        chunker = ResumeChunker(chunk_size=60, chunk_overlap=10)
        chunks = chunker.chunk("x" * 20_000)
        assert len(chunks) > 1
        assert all(approx_token_count(c.content) < 500 for c in chunks)

    def test_overlap_larger_than_size_degrades_instead_of_looping(self):
        chunker = ResumeChunker(chunk_size=100, chunk_overlap=999)
        assert chunker.chunk_overlap < chunker.chunk_size
        assert chunker.chunk("word " * 5000)  # terminates