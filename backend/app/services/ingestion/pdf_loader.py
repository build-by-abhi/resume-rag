"""
File -> plain text.

We keep the loader deliberately small and dependency-light:
* PDF  -> `pypdf` first (pure Python, no binaries), `pdfminer.six` as fallback
* DOCX -> `python-docx`
* TXT  -> direct read

Why not PyMuPDF? It is faster and extracts layout better, but it is a binary
wheel. `pypdf` installs everywhere, which matters for a first project. The
extraction quality gap is handled by `clean_text`, not by a heavier library.
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentParseError(RuntimeError):
    """Raised when a file cannot be turned into text (encrypted, corrupt, ...)."""


def parse_file(path: str | Path) -> str:
    """Dispatch on extension and return raw (uncleaned) text."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    raise DocumentParseError(f"No parser registered for '{suffix}'")


def parse_pdf(path: str | Path) -> str:
    path = Path(path)

    try:
        text = _parse_pdf_pypdf(path)
        if len(text.strip()) > 50:
            # pypdf returns short garbage for some scanned/image-only PDFs.
            return text
        logger.debug("pypdf extracted little text from %s; trying pdfminer", path.name)
    except Exception as exc:
        logger.debug("pypdf failed on %s: %s", path.name, exc)
        text = ""

    try:
        miner_text = _parse_pdf_pdfminer(path)
        if len(miner_text.strip()) > len(text.strip()):
            return miner_text
    except Exception as exc:
        logger.debug("pdfminer failed on %s: %s", path.name, exc)

    if not text.strip():
        raise DocumentParseError(
            f"No text layer found in '{path.name}'. "
            "If it is a scanned image, run OCR first (e.g. OCRmyPDF)."
        )
    return text


def _parse_pdf_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        # Many "encrypted" PDFs are just permission-locked and open with an
        # empty password; try that before giving up.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentParseError(
                f"'{path.name}' is password protected"
            ) from exc

    pages: list[str] = []
    for page in reader.pages:
        # Pages come out with a trailing newline each, which keeps the chunker
        # from gluing the last line of page N to the first line of page N+1.
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _parse_pdf_pdfminer(path: Path) -> str:
    from pdfminer.high_level import extract_text

    return extract_text(str(path)) or ""


def parse_docx(path: str | Path) -> str:
    path = Path(path)
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError("DOCX support needs `python-docx` installed") from exc

    document = docx.Document(str(path))
    blocks = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    # Tables carry contact details in some resume templates.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n".join(blocks)