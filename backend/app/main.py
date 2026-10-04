"""
FastAPI application factory.

`create_app()` is separate from the module-level `app` so tests can build an
isolated instance (custom settings, no background startup) instead of patching
globals.

Startup order matters:
  1. logging
  2. database + create tables (idempotent)
  3. warm the embedding provider so the first real request is not slow
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes import candidates as candidates_routes
from app.api.routes import health as health_routes
from app.api.routes import search as search_routes
from app.api.routes import upload as upload_routes
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.runtime import configure_event_loop
from app.db.session import close_engine, init_models

# Before anything creates a connection (see the Windows note in runtime.py).
configure_event_loop()

logger = get_logger(__name__)

DESCRIPTION = """
Resume RAG + HR search.

Resume search is **not** pure vector search, so this API has three layers:

| Layer | Endpoint | What it does |
|-------|----------|--------------|
| Structured | `GET /api/candidates` | exact filters on name/location/experience/skills |
| Hybrid | `POST /api/search` | filters -> BM25 + vector -> RRF fusion -> rerank |
| Generation | `POST /api/search` (answer) | cited answer, grounded in retrieved chunks |

Typical flow: `POST /api/candidates/upload` then `POST /api/search`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    setup_logging()
    logger.info(
        "starting %s (%s) | embeddings=%s | llm=%s",
        settings.app_name,
        settings.environment,
        settings.embedding_provider,
        settings.llm_provider,
    )

    try:
        await init_models()
        logger.info("database ready at %s", settings.postgres_host)
    except Exception as exc:
        # Do NOT crash: /health must stay reachable so the problem is visible.
        logger.error("database initialisation failed: %s", exc)

    # Load the embedding model up front so the first search is not 10x slower.
    try:
        from app.services.embeddings import get_embedder

        get_embedder()
    except Exception as exc:
        logger.warning("embedding provider warm-up failed: %s", exc)

    yield

    await close_engine()
    logger.info("shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # The React dev server runs on a different port, so CORS is required.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_error(_request, exc: Exception):
        """
        One JSON shape for every unhandled error.

        Returning the traceback to a client is a Phase 5 security problem; in
        development the message is enough, in production we only send a code.
        """
        logger.exception("unhandled error: %s", exc)
        detail = (
            f"{type(exc).__name__}: {exc}"
            if settings.debug
            else "Internal server error. Check the server logs."
        )
        return JSONResponse(status_code=500, content={"detail": detail, "code": "internal_error"})

    # Routers. Health first so it answers even if another router fails to import.
    app.include_router(health_routes.router)
    app.include_router(upload_routes.router)
    app.include_router(candidates_routes.router)
    app.include_router(search_routes.router)

    @app.get("/", tags=["health"])
    async def root() -> dict:
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "endpoints": {
                "upload": "POST /api/candidates/upload",
                "paste_resume": "POST /api/candidates/text",
                "browse": "GET /api/candidates",
                "search": "POST /api/search",
                "jd_match": "POST /api/match",
                "health": "GET /health",
            },
        }

    @app.get("/health/ready", tags=["health"])
    async def ready() -> JSONResponse:
        """Strict readiness: fails when the database is unreachable."""
        from app.db.session import engine

        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"detail": f"database not ready: {exc}", "code": "not_ready"},
            )
        return JSONResponse(status_code=200, content={"status": "ready"})

    # Quieten uvicorn's access log in tests.
    logging.getLogger("uvicorn.access").setLevel(
        logging.INFO if settings.environment == "development" else logging.WARNING
    )
    return app


app = create_app()