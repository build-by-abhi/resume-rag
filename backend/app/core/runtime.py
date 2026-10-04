"""
Process-level runtime setup.

Windows + async psycopg
-----------------------
psycopg3's async mode uses `add_reader`/`add_wait`, which the default Windows
event loop (`ProactorEventLoop`) does not implement. The fix is to use a
`SelectorEventLoop` instead.

There are two separate mechanisms, and only the second is sufficient on its own:

1. `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` - this only
   affects loops created via `asyncio.new_event_loop()`.
2. An explicit loop factory passed to the ASGI server. `uvicorn` builds its loop
   directly and ignores the policy (see `uvicorn/loops/asyncio.py`), so this is
   what actually fixes it. Use `app/run.py`, which does it for you.

Both are set up here so `asyncio.run(...)` in scripts and tests works too.
"""

from __future__ import annotations

import asyncio
import sys

_configured = False


def configure_event_loop() -> None:
    """Select a compatible event loop policy (idempotent)."""
    global _configured
    if _configured:
        return

    if sys.platform == "win32":
        policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy is not None:
            asyncio.set_event_loop_policy(policy())

    _configured = True


def selector_loop_factory():
    """
    Loop factory for an ASGI server / `asyncio.run`.

        uvicorn.run(app, loop_factory=selector_loop_factory())

    Returns None off Windows so uvicorn keeps its own default behaviour.
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop
    return None


configure_event_loop()