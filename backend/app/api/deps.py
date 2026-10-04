"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session

# Annotated alias: routes then just declare `session: SessionDep`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[type(settings), Depends(lambda: settings)]


def require_api_key(x_api_key: str | None = None) -> bool:
    """
    Phase 5 hook: API-key gate.

    When `API_KEY` is set in the environment, every request must present it in
    the `X-API-Key` header. When it is unset the API is open, which is what you
    want for local development.
    """
    expected = getattr(settings, "api_key", None)
    if not expected:
        return True
    return x_api_key == expected