"""
LLM-assisted structured extraction (optional).

RULE FIRST, LLM SECOND. This module never runs before `RuleExtractor`; the
normalizer merges the two and prefers rule values, because regex on a contact
block is more trustworthy than a language model.

When is the LLM actually worth it?
- Resumes with unusual layouts, tables, or two-column PDFs.
- Summaries and seniority that rules cannot infer.
- Skills with no taxonomy entry.

Cost note: one call per resume at ingest. Batch it and cache by `source_hash`
so re-ingesting never re-pays.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.logging import get_logger
from app.prompts.templates import build_extraction_prompt
from app.services.extraction.rule_extractor import Extraction
from app.services.extraction.skills import (
    category_of,
    display_name,
    normalize_skill,
)
from app.services.llm.client import LLMClient, NullLLMClient, get_llm_client

logger = get_logger(__name__)

_MAX_SKILLS = 40


class LLMExtractor:
    """Turns resume text into the same `Extraction` shape the rules produce."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> LLMClient:
        # Resolved lazily so tests can inject a stub.
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    @property
    def enabled(self) -> bool:
        return not isinstance(self.client, NullLLMClient)

    async def extract(self, text: str) -> Extraction:
        if not self.enabled:
            return Extraction()

        system, user = build_extraction_prompt(text)
        try:
            payload = await self.client.complete_json(system, user)
        except Exception as exc:  # never let ingestion die because of the LLM
            logger.warning("LLM extraction failed, continuing with rules only: %s", exc)
            return Extraction()

        if not isinstance(payload, dict):
            return Extraction()
        return self._to_extraction(payload)

    # -- mapping ------------------------------------------------------------

    def _to_extraction(self, data: dict[str, Any]) -> Extraction:
        """
        Parse defensively: LLM output is untrusted input. Every field goes
        through a type check before it can reach the database.
        """
        out = Extraction()

        out.full_name = self._clean_str(data.get("full_name"), 120)
        out.email = self._clean_str(data.get("email"), 320)
        out.phone = self._clean_str(data.get("phone"), 64)
        out.location = self._clean_str(data.get("location"), 160)
        out.current_title = self._clean_str(data.get("current_title"), 200)
        out.current_company = self._clean_str(data.get("current_company"), 200)
        out.summary = self._clean_str(data.get("summary"), 1200)

        years = data.get("years_of_experience")
        if isinstance(years, (int, float)) and 0 <= float(years) <= 60:
            out.total_years_experience = float(years)
        elif isinstance(years, str):
            m = re.search(r"(\d{1,2}(?:\.\d)?)", years)
            if m and 0 <= float(m.group(1)) <= 60:
                out.total_years_experience = float(m.group(1))

        edu = data.get("education")
        if isinstance(edu, dict):
            out.education = {
                k: self._clean_str(edu.get(k), 120)
                for k in ("level", "field", "institution")
                if edu.get(k)
            }

        links = data.get("links")
        if isinstance(links, dict):
            out.links = {k: self._clean_str(v, 300) for k, v in links.items() if v}

        langs = data.get("languages")
        if isinstance(langs, list):
            out.languages = [s.strip()[:40] for s in langs if isinstance(s, str) and s.strip()][:10]

        skills = data.get("skills")
        if isinstance(skills, list):
            for item in skills[:_MAX_SKILLS]:
                if not isinstance(item, str):
                    continue
                surface = item.strip()
                if not 1 < len(surface) <= 60:
                    continue
                canonical = normalize_skill(surface) or surface.lower()
                if canonical not in out.skills:
                    out.skills[canonical] = {
                        "display": (
                            display_name(canonical)
                            if normalize_skill(surface)
                            else surface[:60]
                        ),
                        "category": category_of(canonical) if normalize_skill(surface) else "other",
                        "occurrences": 1,
                        "is_core": True,
                    }
        return out

    @staticmethod
    def _clean_str(value: Any, max_len: int) -> str | None:
        if not isinstance(value, str):
            return None
        value = re.sub(r"\s+", " ", value).strip()
        if not value or value.lower() in {"null", "none", "n/a", "unknown", "-"}:
            return None
        return value[:max_len]


llm_extractor = LLMExtractor()