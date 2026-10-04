"""
Rule-based (regex + heuristics) structured extraction.

This is the deterministic backbone of the project: it always runs, needs no API
key, costs nothing, and is fully unit-testable. The optional LLM extractor runs
*afterwards* and only fills gaps the rules missed -- that ordering is what makes
the pipeline resilient (see `normalizer.py`).

Extracted fields map 1:1 to the structured columns on `candidates`, which is
exactly what enables exact filtering later ("location = Berlin").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from app.core.logging import get_logger
from app.services.extraction.skills import (
    category_of,
    display_name,
    extract_skills,
    normalize_skill,
)
from app.utils.text import extract_urls, normalize_space

# "EXPERIENCE" / "WORK EXPERIENCE" starts the work section.
_EXPERIENCE_HEADER = re.compile(
    r"^(work\s+|professional\s+|relevant\s+)?(experience|employment)\b", re.IGNORECASE
)
# Any of these ends the work section when seen on its own line.
_STOP_SECTION_LINE = re.compile(
    r"^(education|skills|projects|certifications|languages)\b", re.IGNORECASE
)


def _collapse_ws(value: str) -> str:
    """Whitespace tidy-up that preserves case (unlike `normalize_space`)."""
    return re.sub(r"\s+", " ", value or "").strip()

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns (compiled once at import - these run per resume)
# ---------------------------------------------------------------------------
_EMAIL_IN_TEXT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
# Segment shape is `\d{2,4}` then 1-3 more groups. Two deliberate choices:
#   * the separator is REQUIRED between groups, so a bare digit run never merges
#   * the last group allows up to 7 digits, because "5551234" is one chunk, not
#     "5551" + "234" with a phantom separator
# `clean_phone` then validates the digit count and rejects year ranges.
_PHONE_IN_TEXT = re.compile(
    r"(?<![\w.])"
    r"(?:\+\d{1,3}[\s.\-]?)?"
    r"(?:\(\d{1,4}\)[\s.\-]?)?"
    r"\d{2,4}(?:[\s.\-]\d{2,7}){1,3}"
    r"(?![\d])"
)
# "2017 - 2021" has 8 digits and passes a naive phone check. Reject it.
_YEAR_RANGE = re.compile(r"^\d{4}\s*[-–—]\s*\d{4}$")
_URL_IN_TEXT = re.compile(r"(?:https?://|www\.)[^\s,;)]+", re.IGNORECASE)
_GITHUB = re.compile(r"(?:github\.com/|github\.com/)([A-Za-z0-9_.-]+)", re.IGNORECASE)
_LINKEDIN = re.compile(r"linkedin\.com/in/([A-Za-z0-9%\-_.]+)", re.IGNORECASE)

_NAME_LINE = re.compile(r"^([A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+){1,3})\s*$")

# Education: "B.Sc. in Computer Science", "MS Computer Science - MIT",
# "Bachelor of Science, Physics"
_DEGREE_PATTERNS = [
    (r"\b(ph\.?d|doctorate|doctoral)\b", "PhD"),
    (r"\bm\.?sc\.?|\bmasters?\b|\bms\b|\bm\.?s\.?c\.?i?\.?\b", "MS"),
    (r"\bm\.?eng\b|\bmasters? of engineering\b", "MEng"),
    (r"\bm\.?b\.?a\.?|\bmasters? of business\b", "MBA"),
    (r"\bb\.?sc\.?|\bbachelor\b|\bbsc?\b|\bb\.?s\.?\b|\bundergraduate\b", "BSc"),
    (r"\bb\.?eng\b|\bbachelor? of engineering\b|\bb\.?e\b", "BEng"),
    (r"\bm\.?tech\b|\bmasters of technology\b", "MTech"),
    (r"\ba\.?a\.?\s|\bassociate degree\b", "Associate"),
    (r"\bhigh school\b|\bsecondary school\b|\bgcse\b|\babitur\b", "HighSchool"),
    (r"\bdiploma\b|\bcertification\b", "Diploma"),
]

_SENIORITY_RULES = [
    (r"\b(chief|cto|ceo|cio|vp|vice\s+president|head\s+of)\b", "Executive"),
    (r"\b(director|architect|principal|distinguished|fellow)\b", "Principal"),
    (r"\b(manager|lead|leader|supervisor)\b", "Lead"),
    (r"\b(senior|sr\.?|snr|staff)\b", "Senior"),
    (r"\b(junior|jr\.?|associate|entry[- ]level|intern|trainee|graduate)\b", "Junior"),
]

_LOCATION_LINE = re.compile(
    r"^(?:location|based in|address|addressee)\s*[:\-]\s*(.+)$", re.IGNORECASE
)
_CITY_COUNTRY = re.compile(
    r"^([A-Z][a-zA-Z.\- ]{1,28}),\s*([A-Z][a-zA-Z.\- ]{1,28}(?:\s+[A-Z][a-zA-Z.\-]{1,20})*)$"
)
_YEARS_RANGE = re.compile(
    r"(19[5-9]\d|20[0-4]\d)\s*(?:-|–|—|to|until)\s*(present|current|now|19[5-9]\d|20[0-4]\d)",
    re.IGNORECASE,
)
_TOTAL_YEARS = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b.{0,24}?(?:experience|exp\b)",
    re.IGNORECASE,
)
_TITLE_KEYWORDS = re.compile(
    r"\b(engineer|developer|developer|programmer|architect|analyst|scientist|manager|"
    r"consultant|designer|administrator|engineer|lead|head|director|specialist|"
    r"developer|devops|sre|qa|tester|accountant|recruiter|marketer)\b",
    re.IGNORECASE,
)
_STOP_TITLES = re.compile(r"^(curriculum vitae|cv|resume|profile|summary|contact)$", re.IGNORECASE)

# A line that is nothing but a section label ("SKILLS", "TECHNICAL SKILLS:").
_STANDALONE_SKILLS_HEADER = re.compile(
    r"^(technical\s+skills|core\s+competencies|skills|technologies|tech\s+stack)"
    r"\s*[:\-]?\s*$",
    re.IGNORECASE,
)
# Any other standalone section label, used as a boundary when leaving a section.
_STANDALONE_SECTION_HEADER = re.compile(
    r"^("
    r"(professional\s+|work\s+|relevant\s+|technical\s+)?"
    r"(summary|profile|about\s+me|objective|overview|experience|employment|"
    r"work\s+history|career|projects?|portfolio|education|academic\s+\w+|academics|"
    r"qualifications|certifications?|licen[cs]es?|credentials|languages?|"
    r"publications?|research|awards?|honors?|achievements?|activities|interests|"
    r"hobbies|volunteering|references?)"
    r")\s*[:\-]?\s*$",
    re.IGNORECASE,
)


@dataclass
class Extraction:
    """Structured fields pulled out of one resume. All optional by design."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    total_years_experience: float | None = None
    seniority: str | None = None
    summary: str | None = None
    education: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)
    languages: list[str] = field(default_factory=list)
    # {canonical_skill: {"display": str, "occurrences": int, "category": str}}
    skills: dict[str, dict] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any([self.full_name, self.email, self.phone, self.skills])


def clean_phone(raw: str) -> str | None:
    """
    Keep the original formatting but validate the digit count.

    7-15 digits covers national and international formats. Three rejections
    matter in practice:
      * fewer than 7 digits is a date, a zip code or a fragment
      * "2017 - 2021" has 8 digits but is an employment range, not a phone
      * "1111111111" is padding, not a number
    """
    if not raw:
        return None
    candidate = raw.strip()
    if _YEAR_RANGE.match(candidate):
        return None
    digits = re.sub(r"\D", "", candidate)
    if not 7 <= len(digits) <= 15:
        return None
    if len(set(digits)) <= 2:
        return None
    return candidate


class RuleExtractor:
    """Deterministic extraction. No network, no model, no cost."""

    def extract(self, text: str) -> Extraction:
        text = text or ""
        lines = [ln.strip() for ln in text.split("\n")]
        head = "\n".join(lines[:25])  # identity info lives at the top

        out = Extraction()

        # -- contact block ---------------------------------------------------
        out.email = self._first_match(_EMAIL_IN_TEXT, head) or self._first_match(
            _EMAIL_IN_TEXT, text
        )
        out.phone = self._first_phone(head) or self._first_phone(text)
        out.links = self._extract_links(text)

        # -- name ------------------------------------------------------------
        out.full_name = self._extract_name(lines, head)
        out.location = self._extract_location(lines, head, out.links)

        # -- current role ----------------------------------------------------
        out.current_title, out.current_company = self._extract_current_role(lines, text)
        out.seniority = self._extract_seniority(out.current_title, text)

        # -- experience ------------------------------------------------------
        out.total_years_experience = self._estimate_years(text)

        # -- summary ---------------------------------------------------------
        out.summary = self._extract_summary(text)

        # -- education -------------------------------------------------------
        out.education = self._extract_education(lines, text)

        # -- skills ----------------------------------------------------------
        out.skills = self._extract_skills(text)

        logger.debug(
            "rule extraction: name=%s email=%s skills=%d",
            out.full_name,
            out.email,
            len(out.skills),
        )
        return out

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
        m = pattern.search(text or "")
        return m.group(0).strip() if m else None

    def _first_phone(self, text: str) -> str | None:
        for match in _PHONE_IN_TEXT.finditer(text or ""):
            candidate = clean_phone(match.group(0))
            if candidate:
                return candidate
        return None

    def _extract_links(self, text: str) -> dict[str, str]:
        links: dict[str, str] = {}
        urls = [u.rstrip(".,)") for u in extract_urls(text)]
        for url in urls[:5]:
            low = url.lower()
            if "linkedin.com" in low:
                links["linkedin"] = url if low.startswith("http") else f"https://{url}"
            elif "github.com" in low:
                links["github"] = url if low.startswith("http") else f"https://{url}"
            elif "gitlab.com" in low:
                links["gitlab"] = url if low.startswith("http") else f"https://{url}"
            else:
                links.setdefault("website", url if low.startswith("http") else f"https://{url}")

        # Bare handles without a scheme, e.g. "github.com/jane-doe"
        if "github" not in links and (m := _GITHUB.search(text or "")):
            links["github"] = f"https://github.com/{m.group(1)}"
        if "linkedin" not in links and (m := _LINKEDIN.search(text or "")):
            links["linkedin"] = f"https://linkedin.com/in/{m.group(1)}"
        return links

    def _extract_name(self, lines: list[str], head: str) -> str | None:
        """
        Resume names are almost always the first title-cased line, but a
        "Senior Data Engineer" heading sometimes comes first. We validate:
        2-4 words, no digits, no '@', not a job title, not a section header.
        """
        for line in lines[:12]:
            candidate = line.strip(" |-•\t")
            if not candidate or len(candidate) > 60:
                continue
            if "@" in candidate or any(ch.isdigit() for ch in candidate):
                continue
            if _STOP_TITLES.match(candidate) or _TITLE_KEYWORDS.search(candidate):
                continue
            m = _NAME_LINE.match(candidate)
            if m:
                name = m.group(1).strip()
                # Reject all-caps section banners like "PROFESSIONAL SUMMARY".
                if not name.isupper():
                    return name
        return None

    def _extract_location(self, lines: list[str], head: str, links: dict) -> str | None:
        # 1. explicit "Location: Berlin, Germany"
        for line in lines[:20]:
            if m := _LOCATION_LINE.match(line.strip()):
                value = m.group(1).strip(" .,")
                if 2 < len(value) <= 80:
                    return value

        # 2. a bare "City, Country" line
        for line in lines[:15]:
            if m := _CITY_COUNTRY.match(line.strip()):
                return f"{m.group(1).strip()}, {m.group(2).strip()}"

        # 3. fall back to the country part of a LinkedIn URL
        #    (linkedin.com/in/jane -> nothing useful, so we skip)
        return None

    def _extract_current_role(self, lines: list[str], text: str) -> tuple[str | None, str | None]:
        """
        The most recent role is the first entry under an Experience header.
        Formats we handle:
            Senior Backend Engineer | Acme Corp | 2021 - Present
            Acme Corp - Senior Backend Engineer
            Engineer, Acme Corp (Mar 2021 - Present)
        """
        in_experience = False
        for raw_line in lines:
            line = raw_line.strip(" |-–—\t")
            if not line:
                continue
            if _EXPERIENCE_HEADER.match(line):
                in_experience = True
                continue
            if in_experience:
                if _STOP_SECTION_LINE.match(line):
                    break
                # A line that carries a date range is almost certainly a job.
                if (_YEARS_RANGE.search(line) or _TITLE_KEYWORDS.search(line)) and len(line) <= 120:
                    return self._split_title_company(line)

        # Fallback: the first line in the document that looks like a role.
        for line in lines[:12]:
            if _TITLE_KEYWORDS.search(line) and len(line) < 90:
                return self._split_title_company(line.strip())
        return None, None

    def _split_title_company(self, line: str) -> tuple[str | None, str | None]:
        """
        Pull (title, company) out of one line by splitting on the separators
        resumes actually use, then keeping whichever side is the role.
        """
        line = re.sub(
            r"(19[5-9]\d|20[0-4]\d)\s*(?:-|–|—|to)\s*(present|current|now|19[5-9]\d|20[0-4]\d).*$",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip(" |,-–—")

        parts = [p.strip() for p in re.split(r"\s*[|•·]\s*|\s{3,}|,\s{2,}", line) if p.strip()]
        if len(parts) >= 2:
            title = next((p for p in parts if _TITLE_KEYWORDS.search(p)), parts[0])
            company = next(
                (p for p in parts if p is not title and not _TITLE_KEYWORDS.search(p)),
                parts[1],
            )
            return self._tidy_title(title), self._tidy_company(company)
        if parts:
            single = parts[0]
            if re.match(r"^[\w&.\- ]{2,40},\s+[\w&.\- ]{2,40}$", single):
                left, right = single.split(",", 1)
                if _TITLE_KEYWORDS.search(left):
                    return self._tidy_title(left), self._tidy_company(right)
                return self._tidy_title(right), self._tidy_company(left)
            return self._tidy_title(single), None
        return None, None

    @staticmethod
    def _tidy_title(value: str) -> str | None:
        """
        Collapse whitespace but KEEP the original casing.

        Job titles are stored in mixed case ("Staff Backend Engineer") because
        that is how HR filters read them. We only re-case an all-caps line,
        which is a PDF-font artefact rather than a title the person wrote.
        """
        value = _collapse_ws(value)
        if value and value.isupper():
            value = value.title()
        return value[:200] or None

    @staticmethod
    def _tidy_company(value: str) -> str | None:
        return _collapse_ws(value)[:200] or None

    def _extract_seniority(self, title: str | None, text: str) -> str | None:
        """
        Seniority from the current title first, then the top of the document.

        A resume often says "Frontend Developer" for the current job while an
        earlier entry says "Junior Web Developer" -- so when the title carries no
        seniority marker we widen the search to the first ~1500 characters before
        falling back to inferring it from total years.
        """
        for pattern, label in _SENIORITY_RULES:
            if title and re.search(pattern, title, re.IGNORECASE):
                return label
        for pattern, label in _SENIORITY_RULES:
            if re.search(pattern, text[:1500], re.IGNORECASE):
                return label

        years = self._estimate_years(text)
        if years is not None:
            if years >= 10:
                return "Senior"
            if years >= 5:
                return "Mid"
        return None

    def _estimate_years(self, text: str) -> float | None:
        """
        Two strategies, most trustworthy first:
        1. explicit claim ("8+ years of experience")
        2. sum of non-overlapping employment date ranges (handles career gaps)
        """
        if m := _TOTAL_YEARS.search(text or ""):
            try:
                return min(50.0, float(m.group(1)))
            except ValueError:
                pass

        ranges: list[tuple[int, int]] = []
        today = date.today().year
        for start_raw, end_raw in _YEARS_RANGE.findall(text or ""):
            start = int(start_raw)
            end = today if end_raw.lower() in {"present", "current", "now"} else int(end_raw)
            if 1970 <= start <= today + 1 and 1970 <= end <= today + 1 and end >= start:
                ranges.append((start, end))

        if not ranges:
            return None

        # Merge overlapping ranges so a job listed twice is not double counted.
        ranges.sort()
        merged: list[list[int]] = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        # Date ranges are year-granular, so subtract 1 per range to avoid
        # over-counting ("2020 - 2022" should be ~2, not 3).
        total = sum(max(0, end - start - 1) for start, end in merged)
        return float(total) if total > 0 else None

    def _extract_summary(self, text: str) -> str | None:
        """
        Prefer an explicit Summary/Profile section; otherwise take the first
        2-3 sentences of the document.
        """
        m = re.search(
            r"(?:summary|profile|about(?:\s+me)?|objective)\s*[:\-]?\s*\n(.{80,900}?)"
            r"(?=\n\s*\n\s*(?:experience|work|education|skills|projects)|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            summary = normalize_space(m.group(1))
            if 40 <= len(summary) <= 900:
                return summary

        # Fallback: first paragraph that is not the name/contact block.
        for para in text.split("\n\n")[:3]:
            candidate = normalize_space(para)
            if (
                60 <= len(candidate) <= 700
                and not _EMAIL_IN_TEXT.search(candidate)
                and not _NAME_LINE.match(candidate.strip())
            ):
                return candidate
        return None

    def _extract_education(self, lines: list[str], text: str) -> dict:
        """Return {"level": "BSc", "field": ..., "institution": ...}."""
        out: dict = {}
        # `universit` deliberately has no trailing boundary: clean_text strips accents,
        # so "Universität" becomes "Universitat", and `\buniversit\b` would no
        # longer match.
        institution_re = r"\b(universit\w*|college|institute|polytechnic|academy)\b"
        degree_re = (
            r"\b(universit\w*|college|institute|school|polytechnic|b\.?sc|m\.?sc|bachelor|"
            r"master|ph\.?d|b\.?tech|m\.?tech|b\.?eng|m\.?eng|b\.?e|degree|gpa)\b"
        )
        candidates = [ln for ln in lines if re.search(degree_re, ln, re.IGNORECASE)]
        blob = " \n".join(candidates[:8]) if candidates else text[:1500]

        for pattern, level in _DEGREE_PATTERNS:
            if re.search(pattern, blob, re.IGNORECASE):
                out["level"] = level
                break

        # "MSc in Computer Science" / "Bachelor of Science, Physics".
        # The degree keyword is REQUIRED before "in"/"of". Without it, the
        # preposition inside "University of Cape Town" matches and the field
        # becomes "Cape Town" instead of "Software Engineering".
        if (m := re.search(
            r"\b(?:degree|bachelor|master|b\.?sc|m\.?sc|b\.?tech|m\.?tech|"
            r"ph\.?d|b\.?eng|m\.?eng|m\.?a|b\.?a|diploma|programme|program)"
            r"\.?\s+(?:in|of)\s+([A-Za-z&.\- ]{3,50})",
            blob,
            re.IGNORECASE,
        )) or (m := re.search(
            # Fallback for "B.Sc. Computer Science" (no preposition at all).
            # `\.?` after the abbreviation is required: the line reads
            # "B.Eng. Software Engineering", so without it the trailing dot
            # blocks the whitespace match and no field is found at all.
            r"\b(?:b\.?sc|m\.?sc|b\.?tech|m\.?tech|ph\.?d|b\.?eng|m\.?eng|b\.?e)\.?"
            r"\s+([A-Za-z&.\- ]{3,50})",
            blob,
            re.IGNORECASE,
        )):
            field = _collapse_ws(m.group(1))
            if field and not re.search(institution_re, field, re.IGNORECASE):
                out["field"] = field.title()[:80]

        # Institution: take just the segment that contains the keyword, so we
        # store "University of Cape Town" instead of the whole pipe-delimited
        # education line.
        for line in candidates:
            if not re.search(institution_re, line, re.IGNORECASE):
                continue
            for segment in re.split(r"\s*[|｜]\s*", line):
                if re.search(institution_re, segment, re.IGNORECASE):
                    out["institution"] = _collapse_ws(segment)[:120]
                    break
            else:
                out["institution"] = _collapse_ws(line)[:120]
            break

        if m := re.search(r"\bGPA\s*[:\-]?\s*(\d\.\d)", text, re.IGNORECASE):
            out["gpa"] = float(m.group(1))
        return out

    def _extract_skills(self, text: str) -> dict[str, dict]:
        """
        Two passes:
        1. taxonomy scan over the whole document -> canonical skills + counts
        2. dedicated Skills-section parse -> picks up technologies we do not
           have in the taxonomy (stored as their own lowercase key)
        """
        counts = extract_skills(text)
        result: dict[str, dict] = {
            key: {
                "display": display_name(key),
                "category": category_of(key),
                "occurrences": count,
                "is_core": False,
            }
            for key, count in counts.items()
        }

        # Skills section: reward explicit mentions as "core skills".
        section = self._find_skills_section(text)
        if section:
            for surface in self._iter_section_tokens(section):
                canonical = normalize_skill(surface)
                if canonical:
                    entry = result.setdefault(
                        canonical,
                        {
                            "display": display_name(canonical),
                            "category": category_of(canonical),
                            "occurrences": 0,
                            "is_core": True,
                        },
                    )
                    entry["occurrences"] += 2  # explicit listing weighs more
                    entry["is_core"] = True
                else:
                    key = surface.strip().lower()
                    if 1 < len(key) <= 40:
                        entry = result.setdefault(
                            key,
                            {
                                "display": surface.strip()[:40],
                                "category": "other",
                                "occurrences": 0,
                                "is_core": True,
                            },
                        )
                        entry["occurrences"] += 2
                        entry["is_core"] = True
        return result

    @staticmethod
    def _find_skills_section(text: str) -> str | None:
        """
        Return the body of the Skills section, or None.

        Implemented line-by-line rather than as one big regex because skills
        blocks contain lines that *look* like section headers ("Languages: Python"
        starts with "Languages"). A regex cannot tell "Languages: Python" (part of
        the skills block) from a real "Languages" header without a blank line in
        front of it; walking the lines makes that distinction explicit.
        """
        lines = (text or "").split("\n")
        start = None
        inline_tail = ""

        for index, line in enumerate(lines):
            stripped = line.strip()
            # Inline form: "Skills: Python, Docker, AWS" -> take the tail.
            if re.match(r"^\s*(technical\s+)?skills\b\s*[:\-]\s*\S", stripped, re.IGNORECASE):
                inline_tail = stripped.split(":", 1)[1]
                start = index + 1
                break
            # Standalone header form: the whole line is just the label.
            if _STANDALONE_SKILLS_HEADER.match(stripped):
                start = index + 1
                break

        if start is None:
            return None
        if inline_tail:
            return inline_tail

        collected: list[str] = []
        for line in lines[start : start + 15]:
            stripped = line.strip()
            # A real section header ends the block. It is only a header if the
            # previous line was blank (that is what separates sections).
            if not stripped:
                if collected:
                    break
                continue
            if collected and _STANDALONE_SECTION_HEADER.match(stripped):
                break
            if not collected and _STANDALONE_SECTION_HEADER.match(stripped):
                break
            collected.append(stripped)

        return "\n".join(collected) or None

    @staticmethod
    def _iter_section_tokens(section: str) -> list[str]:
        from app.services.extraction.skills import parse_skill_list

        tokens: list[str] = []
        for line in section.split("\n"):
            tokens.extend(parse_skill_list(line))
        return tokens


rule_extractor = RuleExtractor()