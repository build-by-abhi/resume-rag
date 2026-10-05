"""Log configuration shared by the whole app.

WHY THE STREAM IS CONFIGURABLE
------------------------------
Under HTTP, logs on stdout are fine. Under an MCP **stdio** server they are
fatal: stdout IS the JSON-RPC transport, so a single log line is parsed as a
malformed protocol frame and the client fails with

    Response ended unexpectedly and may be incomplete.

The MCP SDK mitigates this - it claims file descriptor 1 for the protocol and
repoints fd 1 at stderr while serving (see `mcp/server/stdio.py`). But that only
happens once `run()` is called. Anything logged BEFORE that point - building the
server, validating config - still lands on real stdout and corrupts the stream.

So the rule is: `python -m app.mcp` configures logging to stderr up front, and
this module never lets a later no-argument `setup_logging()` move it back.
"""

from __future__ import annotations

import logging
import sys
from typing import IO

from app.core.config import settings

_CONFIGURED = False
#: Once chosen, never silently changed back to stdout.
_STREAM: IO[str] = sys.stdout


def setup_logging(stream: IO[str] | None = None) -> None:
    """
    Idempotent root-logger setup so re-imports never duplicate handlers.

    Args:
        stream: where to log. Pass `sys.stderr` for any process whose stdout is
            a protocol channel. An explicit stream always wins; a later call
            without one is a no-op, which prevents `get_logger()` - called
            implicitly from module import - from reverting the choice.
    """
    global _CONFIGURED, _STREAM

    if stream is not None:
        _STREAM = stream
    elif _CONFIGURED:
        return

    handler = logging.StreamHandler(_STREAM)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers = [handler]

    # These are chatty and rarely useful in a portfolio project. Silencing them
    # also removes a second source of accidental stdout noise.
    for noisy in ("httpx", "httpcore", "pdfminer", "aiosqlite", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Shortcut for `logging.getLogger` that guarantees setup ran first."""
    setup_logging()
    return logging.getLogger(name)


def reset_logging_for_tests() -> None:
    """Restore defaults so a test can re-run `setup_logging` from scratch."""
    global _CONFIGURED, _STREAM
    _CONFIGURED = False
    _STREAM = sys.stdout