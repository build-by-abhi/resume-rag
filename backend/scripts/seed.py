"""
Seed the database with sample resumes so the UI has something to show.

    python -m scripts.seed            # ingest the built-in samples
    python -m scripts.seed --reset    # wipe candidates first
    python -m scripts.seed path/to/*.pdf

WHY A SEED SCRIPT
-----------------
It calls the same `IngestionPipeline` the HTTP endpoint uses, so it is not a
second code path that can drift. It is also the fastest way to rebuild a demo
database after changing CHUNK_SIZE or EMBEDDING_DIM (both of which change the
stored vectors and require a re-ingest).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow `python -m scripts.seed` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.core.runtime import configure_event_loop  # noqa: E402

configure_event_loop()

from app.db.session import (  # noqa: E402
    SessionLocal,
    close_engine,
    drop_models,
    embedding_dim_mismatch,
    init_models,
)
from app.services.ingestion.pipeline import pipeline  # noqa: E402

logger = get_logger("seed")

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_resumes"


async def reset(session) -> None:
    """Delete all candidates. Chunk and skill rows cascade."""
    from sqlalchemy import delete

    from app.db.models import Candidate

    result = await session.execute(delete(Candidate))
    await session.commit()
    logger.info("removed %d existing candidates", result.rowcount or 0)


async def ingest_files(paths: list[Path]) -> None:
    await init_models()

    # Catch the vector-width mismatch here, where the message can actually help,
    # instead of letting Postgres fail every INSERT with
    # "expected 1536 dimensions, not 384".
    if mismatch := await embedding_dim_mismatch():
        print(f"\nCONFIGURATION ERROR\n{mismatch}\n")
        await close_engine()
        raise SystemExit(2)

    async with SessionLocal() as session:
        for path in paths:
            try:
                result = await pipeline.ingest_file(session, path)
                candidate = result.candidate
                print(
                    f"  OK   {path.name:<40} "
                    f"{candidate.full_name or 'Unnamed':<24} "
                    f"{result.chunk_count:>3} chunks  "
                    f"{result.skill_count:>3} skills  "
                    f"[{result.candidate.extraction_method}]"
                )
            except Exception as exc:
                # One bad file must not abort the whole seed run. Roll back so
                # the session is usable for the next file - without this, the
                # second file fails with PendingRollbackError instead of its
                # own real error.
                await session.rollback()
                print(f"  FAIL {path.name:<40} {type(exc).__name__}: {exc}")

    await close_engine()


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the resume database.")
    parser.add_argument("paths", nargs="*", type=Path, help="resume files to ingest")
    parser.add_argument(
        "--reset", action="store_true", help="delete existing candidates first"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="DROP all tables first (required after changing EMBEDDING_DIM)",
    )
    args = parser.parse_args()

    setup_logging()

    paths = args.paths or sorted(
        [p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".txt", ".pdf", ".docx"}]
    )
    if not paths:
        print(
            "No files to ingest. Put sample resumes in "
            f"{SAMPLE_DIR} or pass paths explicitly."
        )
        return 1

    print(f"Ingesting {len(paths)} file(s)...")

    if args.drop:

        async def _drop() -> None:
            await drop_models()
            await close_engine()

        print("Dropping all tables...")
        asyncio.run(_drop())

    if args.reset:
        async def _reset() -> None:
            await init_models()
            async with SessionLocal() as session:
                await reset(session)
            await close_engine()

        asyncio.run(_reset())

    asyncio.run(ingest_files(paths))
    print("\nDone. Start the API with:  uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())