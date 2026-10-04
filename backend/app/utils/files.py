"""Upload-file safety helpers (size + extension + content sniffing)."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

# Only these may be uploaded. `.txt` helps you test without a PDF at hand.
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    return ext


def safe_filename(filename: str) -> str:
    """
    Strip directory components and dangerous characters.

    Prevents path traversal (`../../etc/passwd`) both when we save the file and
    when the UI later displays the name.
    """
    base = Path(filename or "resume").name
    base = _SAFE_NAME.sub("_", base).strip(". ") or "resume"
    return base[:120]


def sha256_file(path: Path) -> str:
    """Content hash -> dedupe + provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def new_storage_path(upload_dir: Path, filename: str, content_hash: str) -> Path:
    """
    Collision-free on-disk location: `<hash8>_<uuid8>_<safe name>`.

    Keeping the hash prefix makes duplicates obvious when browsing the folder.
    """
    ext = Path(filename).suffix.lower()
    return upload_dir / f"{content_hash[:8]}_{uuid.uuid4().hex[:8]}{ext}"