"""
Uvicorn entrypoint.

WHY THIS FILE EXISTS (Windows only)
------------------------------------
uvicorn builds its event loop directly rather than honouring
`asyncio.set_event_loop_policy`:

    # uvicorn/loops/asyncio.py
    if sys.platform == "win32" and not use_subprocess:
        return asyncio.ProactorEventLoop

So setting the policy to `WindowsSelectorEventLoopPolicy` (see
`app/core/runtime.py`) is not enough. psycopg3's async mode needs a
SelectorEventLoop, because `ProactorEventLoop` does not implement
`add_reader` / `add_wait`.

Fix: pass an explicit loop factory. On Linux and macOS this returns None and
uvicorn uses its normal behaviour.

Usage
-----
    python -m app.run                 # development
    python -m app.run --reload        # auto-reload
    python -m app.run --port 8080

On Linux/macOS the plain `uvicorn app.main:app --reload` works just as well.
"""

from __future__ import annotations

import argparse

import uvicorn

from app.core.runtime import selector_loop_factory


def _loop_setting():
    """
    The `loop=` argument for uvicorn.

    uvicorn accepts either a dotted string or a callable returning a loop class.
    A callable is the only way to force a SelectorEventLoop on Windows; on other
    platforms "auto" keeps uvicorn's own default (uvloop when installed).
    """
    factory = selector_loop_factory()
    return factory if factory is not None else "auto"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Resume RAG API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on change")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    kwargs = {
        "app": "app.main:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "workers": args.workers if not args.reload else 1,
        "loop": _loop_setting(),
    }
    uvicorn.run(**kwargs)


if __name__ == "__main__":
    main()