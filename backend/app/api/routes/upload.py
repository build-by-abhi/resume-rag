"""
Resume upload (Phase 1).

Upload -> save to disk -> ingest -> candidate + chunks + embeddings in the DB.

Upload safety (all applied here, before anything is parsed):
  1. extension whitelist
  2. size limit enforced while streaming, so a 4 GB file never lands on disk
  3. randomised filename + sanitised original name
  4. sha256 dedupe so re-uploading the same resume does not duplicate a profile

Endpoints
    POST /api/candidates/upload   one or many files
    POST /api/candidates/text     paste raw text (fastest way to try it out)
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import SessionDep
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.candidate import IngestSummary, IngestTextRequest, UploadResponse
from app.services.ingestion.pdf_loader import DocumentParseError
from app.services.ingestion.pipeline import IngestResult, pipeline
from app.utils.files import new_storage_path, safe_filename, sha256_file, validate_extension

router = APIRouter(prefix="/api/candidates", tags=["ingestion"])
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_resumes(
    session: SessionDep, files: list[UploadFile] = File(...)
) -> UploadResponse:
    """
    Accepts one or many files in a single multipart request.

    Each file is processed independently: one bad PDF must not fail the batch.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload at most 20 files per request")

    results: list[IngestSummary] = []
    errors: list[dict[str, str]] = []
    started = time.perf_counter()

    for upload in files:
        filename = safe_filename(upload.filename or "resume")
        try:
            validate_extension(filename)
        except ValueError as exc:
            errors.append({"filename": filename, "error": str(exc)})
            continue

        try:
            # Save first: the ingestion pipeline works from a path, which keeps
            # it reusable from scripts and CLI tools, not just from HTTP.
            path = await _save_upload(upload, filename)
            result = await pipeline.ingest_file(
                session, path, original_filename=filename
            )
            results.append(_to_summary(result))
        except DocumentParseError as exc:
            errors.append({"filename": filename, "error": str(exc)})
        except ValueError as exc:
            errors.append({"filename": filename, "error": str(exc)})
        except Exception as exc:
            logger.exception("failed to ingest %s", filename)
            errors.append({"filename": filename, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            await upload.close()

    logger.info(
        "upload batch: %d ok, %d failed in %.0fms",
        len(results),
        len(errors),
        (time.perf_counter() - started) * 1000,
    )
    return UploadResponse(
        batch_size=len(files),
        succeeded=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )


@router.post("/text", response_model=IngestSummary, status_code=201)
async def ingest_text(session: SessionDep, payload: IngestTextRequest) -> IngestSummary:
    """
    Ingest a pasted resume as text:  POST /api/candidates/text  {"text": "..."}

    Not a toy: it is the fastest loop for iterating on extraction and chunking
    without generating PDF files, and it is how you build the eval set in Phase 5.
    """
    text = payload.text.strip()
    if len(text) < 80:
        raise HTTPException(status_code=422, detail="Resume text is too short (min 80 chars)")

    path = settings.upload_dir / f"pasted_{uuid.uuid4().hex[:12]}.txt"
    path.write_text(text, encoding="utf-8")
    try:
        result = await pipeline.ingest_file(session, path, original_filename="pasted.txt")
    finally:
        path.unlink(missing_ok=True)  # pasted text has no need to persist
    return _to_summary(result)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
async def _save_upload(upload: UploadFile, filename: str):
    """
    Stream the upload to disk in 1 MB blocks, aborting as soon as the size limit
    is exceeded so an oversized upload cannot fill the volume.

    Returns the final `pathlib.Path`.
    """
    limit = settings.max_upload_bytes
    temp_path = settings.upload_dir / f".upload_{uuid.uuid4().hex}.tmp"

    written = 0
    try:
        with temp_path.open("wb") as fh:
            while chunk := await upload.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_mb} MB limit",
                    )
                fh.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Final path: hash-prefixed + uuid so uploads never overwrite each other.
        final_path = new_storage_path(settings.upload_dir, filename, sha256_file(temp_path))
        temp_path.replace(final_path)
        return final_path
    finally:
        temp_path.unlink(missing_ok=True)


def _to_summary(result: IngestResult) -> IngestSummary:
    candidate = result.candidate
    warnings: list[str] = []
    # Surface extraction gaps so the reviewer knows the rules/limits are real.
    # NOTE: the ORM attribute is `skill_links`, not `skills` - `skills` only
    # exists on the API schema.
    if not candidate.full_name:
        warnings.append("Could not detect the candidate's name")
    if not candidate.email:
        warnings.append("No email found")
    if not candidate.skill_links:
        warnings.append("No known skills detected")

    return IngestSummary(
        candidate_id=candidate.id,
        created=result.created,
        chunk_count=result.chunk_count,
        skill_count=result.skill_count,
        text_length=result.text_chars,
        full_name=candidate.full_name,
        extraction_method=candidate.extraction_method,
        warnings=warnings,
    )