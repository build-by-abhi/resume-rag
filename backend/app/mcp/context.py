"""
Database access for the MCP server.

WHY THIS EXISTS
---------------
Every tool and resource needs a database session, but an MCP server has no
FastAPI dependency-injection to lean on. Rather than opening a session per call
(which would exhaust the pool under a burst) this module owns one lazily
created session for the process lifetime.

A MOMENT ON STATELESSNESS
-------------------------
MCP itself is stateless: nothing may be inferred from a previous request. This
class is NOT protocol state - it is a connection pool. Any request can be
handled by any instance, because the pool is per-process and the SQL is
per-request. That is what makes the HTTP transport horizontally scalable.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import SessionLocal, init_models

logger = get_logger(__name__)


class Database:
    """
    Lazily-created, process-wide database session.

    Used as an async context manager so callers cannot forget to release it:

        async with database.session() as session:
            ...
    """

    def __init__(self) -> None:
        self._session: AsyncSession | None = None
        self._ready = False
        # When set, every `session()` call hands back THIS session instead of
        # opening one. Used by the test suite to point the MCP layer at the
        # dedicated test database without standing up a second engine.
        self._forced: AsyncSession | None = None

    def force_session(self, session: AsyncSession) -> None:
        """Pin this instance to an existing session (tests only)."""
        self._forced = session

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        # An MCP stdio server is spawned by the client, so `lifespan` on the
        # FastAPI app never runs. Create the schema here instead.
        await init_models()
        self._ready = True
        logger.info("database ready for MCP server")

    def session(self):
        return _SessionContext(self)

    async def _acquire(self) -> AsyncSession:
        if self._forced is not None:
            return self._forced
        await self._ensure_ready()
        if self._session is None:
            self._session = SessionLocal()
            logger.debug("opened MCP database session")
        return self._session

    async def _release(self, session: AsyncSession) -> None:
        # A forced session belongs to the caller (the test fixture), so leave
        # its transaction lifecycle alone.
        if self._forced is not None:
            return
        # Roll back anything a handler left open. A read-only tool that raised
        # mid-query must not poison the next caller.
        if session.in_transaction():
            await session.rollback()

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
            logger.info("closed MCP database session")


class _SessionContext:
    """Async context manager returned by `Database.session()`."""

    def __init__(self, database: Database) -> None:
        self._db = database
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self._session = await self._db._acquire()
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._db._release(self._session)


#: Process-wide instance. Created here rather than at import so a unit test can
#: swap it before anything touches the database.
database = Database()


async def fetch_all(sql: Any, params: dict | None = None) -> list[dict]:
    """Run a read-only query and return plain dicts.

    Plain dicts (not ORM objects) because MCP results are serialised straight to
    JSON - an ORM instance has no meaningful JSON representation.
    """
    from sqlalchemy import text as sql_text

    async with database.session() as session:
        result = await session.execute(sql_text(sql), params or {})
        return [dict(row) for row in result.mappings().all()]


async def fetch_one(sql: Any, params: dict | None = None) -> dict | None:
    """Same as `fetch_all` but expects at most one row."""
    rows = await fetch_all(sql, params)
    return rows[0] if rows else None


async def count(sql: str, params: dict | None = None) -> int:
    """Run a scalar `SELECT count(*)`-style query."""
    from sqlalchemy import text as sql_text

    async with database.session() as session:
        result = await session.execute(sql_text(sql), params or {})
        return int(result.scalar_one() or 0)