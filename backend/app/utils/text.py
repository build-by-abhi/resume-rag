"""
Text cleaning + normalisation helpers.

PDF extraction is messy: ligatures, broken hyphens, duplicated spaces, soft
line-wraps mid-sentence. Everything downstream (extraction, chunking, search)
assumes clean text, so cleaning happens exactly once, right after parsing.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that often survive PDF extraction but add no meaning.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
# NOTE: `str.translate` keys must be ORDINALS, not characters, so every entry is
# wrapped in `ord(...)`. Without this the replacements silently do nothing.
_LIGATURES = {ord(k): v for k, v in {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
}.items()}

_WS = re.compile(r"[ \t\u00a0]+")
_BLANKS = re.compile(r"\n{3,}")
# "experi-\nence" -> "experience" (hyphenated word split across lines).
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
# Bullet noise we do not want in embeddings.
_BULLETS = re.compile(
    r"^\s*[\u2022\u25aa\u25cf\u2023\u2043\u2219*\-\u2013\u00b7o]\s*", re.MULTILINE
)
# Line numbers / headers / footers that appear on every page.
_PAGE_NOISE = re.compile(r"^\s*(page\s*\d+(\s*of\s*\d+)?|\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
    r"\d{3,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?"
)
_URL = re.compile(
    # Full URLs, plus the bare domains resumes very often print without a scheme.
    r"(?:https?://|www\.)[^\s,;)]+|(?:linkedin\.com|github\.com|gitlab\.com|"
    r"medium\.com|behance\.net|dribbble\.com|stackoverflow\.com)/[^\s,;)]+",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalise text coming out of a PDF."""
    if not text:
        return ""

    # 1. Expand ligatures FIRST. If we ran NFKD+ascii-strip first, the "ﬁ" in
    #    "eﬃcient" would be deleted outright instead of becoming "ffi".
    text = text.translate(_ZERO_WIDTH).translate(_LIGATURES)

    # 2. Unicode normalisation: decompose then drop accents. We keep letters,
    #    so "José" -> "Jose" makes matching more forgiving.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 3. Repair words broken across line wraps.
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # 3. Normalise newlines to \n (PDFs love \r\n and form feeds \f).
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")

    # 4. Strip page furniture.
    text = _PAGE_NOISE.sub("", text)
    text = _BULLETS.sub("", text)

    # 5. Collapse whitespace, but keep paragraph breaks.
    text = _WS.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _BLANKS.sub("\n\n", text)

    return text.strip()


def approx_token_count(text: str) -> int:
    """
    Cheap token estimate (~4 chars/token) so we avoid depending on tiktoken.
    Good enough for chunk sizing and display; exact tokenizers only matter when
    you hit a context limit, which we handle conservatively in the LLM layer.
    """
    return max(1, len(text) // 4) if text else 0


def normalize_space(text: str) -> str:
    """Lowercase + collapse whitespace. Used for keyword/prefix matching."""
    return _WS.sub(" ", text or "").strip().lower()


def extract_emails(text: str) -> list[str]:
    return _EMAIL.findall(text or "")


def extract_phones(text: str) -> list[str]:
    """
    Loose phone detector. Returns candidates verbatim; validity is checked by
    the extractor (we only accept 7-15 digits after stripping separators).
    """
    return _PHONE.findall(text or "")


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,)") for u in _URL.findall(text or "")]


def truncate(text: str, limit: int = 280, suffix: str = "...") -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - len(suffix)].rstrip() + suffix


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter used to build chunk boundaries.

    Splits on . ! ? followed by whitespace + capital, which is good enough for
    resumes and avoids a heavyweight NLP dependency.
    """
    text = normalize_space_keep_newlines(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def normalize_space_keep_newlines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())