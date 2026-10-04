"""Database engine + session helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logging import get_logger
from app.core.runtime import configure_event_loop

logger = get_logger(__name__)

# Must happen before the engine (and therefore any connection) is created.
# On Windows, async psycopg requires a SelectorEventLoop.
configure_event_loop()

# `pool_pre_ping` avoids handing out connections that Postgres closed while idle
# (very common with Docker + free-tier Postgres).
engine = create_async_engine(
    settings.sqlalchemy_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# AsyncSession is the async equivalent of Session. The session is a "unit of
# work": you `add()` objects, then `commit()` to write them in one transaction.
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,  # keep ORM objects usable after commit()
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields one session per request and guarantees
    rollback if the handler raises.
    """
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_models() -> None:
    """
    Create tables if they don't exist.

    Fine for a portfolio/demo app. For production you would switch to Alembic
    migrations (Phase 5) so schema changes are versioned and reversible.
    """
    # Must import models so they are registered on Base.metadata before create_all.
    from app.db import models  # noqa: F401
    from app.db.base import Base

    async with engine.begin() as conn:
        # pgvector/pg_trgm must exist before tables that use `vector`.
        await conn.run_sync(_ensure_extensions)
        await conn.run_sync(Base.metadata.create_all)


async def embedding_dim_mismatch() -> str | None:
    """
    Detect the single most confusing configuration mistake in this project.

    `EMBEDDING_DIM` is baked into the `vector(N)` column when the table is
    created. Change it later and Postgres rejects every insert with
    "expected 1536 dimensions, not 384" - a message that says nothing about the
    actual cause.

    Returns a human-readable explanation when the table and the code disagree,
    otherwise None.
    """
    from sqlalchemy import text as sql_text

    from app.db.models import Candidate, ResumeChunk

    expected_model = ResumeChunk.embedding.type.dim
    if expected_model != settings.embedding_dim:
        return (
            f"EMBEDDING_DIM={settings.embedding_dim} in .env does not match the "
            f"model definition (dim={expected_model}). They are read from the same "
            "setting, so this usually means a stale .env - restart the process."
        )

    try:
        async with engine.connect() as conn:
            for table, column in (
                (Candidate.__tablename__, "profile_embedding"),
                (ResumeChunk.__tablename__, "embedding"),
            ):
                actual = await conn.scalar(
                    sql_text(
                        """
                        SELECT format_type(a.atttypid, a.atttypmod)
                        FROM pg_attribute a
                        JOIN pg_class t ON t.oid = a.attrelid
                        WHERE t.relname = :table AND a.attname = :column
                        """
                    ),
                    {"table": table, "column": column},
                )
                if actual is None:
                    continue
                # format_type returns e.g. 'vector(1536)'
                if actual.startswith("vector") and settings.embedding_dim:
                    declared = (
                        int(actual.removeprefix("vector(").rstrip(")"))
                        if "(" in actual
                        else None
                    )
                    if declared is not None and declared != settings.embedding_dim:
                        return (
                            f"Database column {table}.{column} is {actual} but "
                            f"EMBEDDING_DIM={settings.embedding_dim}. Existing rows "
                            "cannot be read by the new model. Fix it by either "
                            "setting EMBEDDING_DIM back to "
                            f"{declared} and re-ingesting, or by dropping the "
                            "tables and re-ingesting: "
                            "`python -m scripts.seed --drop`."
                        )
    except Exception as exc:  # never block startup on a diagnostic query
        logger.debug("embedding dim check skipped: %s", exc)

    return None


def _ensure_extensions(connection) -> None:
    """
    CREATE EXTENSION IF NOT EXISTS for everything this app relies on.

    NOTE: this must be SYNCHRONOUS. `AsyncConnection.run_sync` hands us a
    synchronous DBAPI connection and runs the callable on a greenlet, so passing
    an `async def` here silently creates a coroutine that is never awaited.
    """
    for ext in ("vector", "pg_trgm"):
        connection.exec_driver_sql(f"CREATE EXTENSION IF NOT EXISTS {ext}")


async def drop_models() -> None:
    """
    Drop every table this app owns. Used by `scripts/seed.py --drop`, which is
    how you recover from changing EMBEDDING_DIM (the vector column width is
    baked in at CREATE TABLE time and cannot be altered).
    """
    from app.db import models  # noqa: F401
    from app.db.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("dropped all tables")


async def close_engine() -> None:
    """Dispose of the pool on shutdown."""
    await engine.dispose()