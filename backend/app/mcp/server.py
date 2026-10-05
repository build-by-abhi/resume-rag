"""
Server assembly and the CLI entrypoint.

Run it:

    # stdio - for Claude Desktop, an IDE, or any local MCP host
    python -m app.mcp

    # Streamable HTTP - remote / shared / multi-tenant
    python -m app.mcp --transport http --host 0.0.0.0 --port 9000

WHAT IS DECLARED HERE
---------------------
`instructions` is server-level text the host injects into the model's system
context. It is the one place to state the rules that apply to EVERY tool call -
what this server is, what it must not be used for, and how to handle the edge
cases that are not per-tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.logging import get_logger, setup_logging

logger = get_logger("mcp")

# The instruction block the host shows the model. Written as operating rules
# rather than prose, because that is what survives contact with a model.
INSTRUCTIONS = """\
This server exposes a searchable pool of candidate resumes.

What each primitive is for:

- `search_candidates`    - who has some experience. Hybrid retrieval over
                           structured filters plus BM25 plus vectors.
- `match_job_description`- who fits a specific role spec.
- `get_candidate_profile` (resource) - everything about one known person.
- `candidate_directory`  (resource) - the whole pool, one line each.
- `skill_facets`         (resource) - which skills actually exist here.
- `corpus_stats`         (resource) - is the pool even populated?
- `ingest_resume`        - add a person. THIS WRITES.

Operating rules:

1. Only claim what the `evidence` excerpts state. A candidate has an experience
   only if a retrieved excerpt says so. Never infer seniority, dates or
   responsibilities that are not written down.

2. Check `corpus_stats` when a search returns nothing. "No matches" and "empty
   database" and "LLM disabled" are different problems with different fixes.

3. Do not call `ingest_resume` to look someone up - it creates rows. Use
   `search_candidates`. Re-submitting identical text is deduplicated, so it is
   safe, but it is still a write.

4. Prefer `search_candidates` over guessing skill names. Call `skill_facets`
   first if you are unsure; an unknown skill returns zero results silently.

5. Candidate data is personal information (name, email, phone, location).
   Return only the fields the user asked for.

6. If nothing matches, say so. A short honest answer beats a padded one.
"""

SERVER_NAME = "resume-rag"
SERVER_VERSION = "0.1.0"
SERVER_TITLE = "Resume RAG - Candidate Search"


def build_server():
    """
    Construct the MCPServer with every tool, resource and prompt registered.

    Returned rather than module-level so a test can build an isolated instance
    (and so importing this module does not start a server).
    """
    from mcp.server import MCPServer

    from app.mcp.prompts import register_prompts
    from app.mcp.resources import register_resources
    from app.mcp.tools import register_tools

    server = MCPServer(
        name=SERVER_NAME,
        title=SERVER_TITLE,
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
        # stdio servers must never log to stdout: stdout IS the transport, so a
        # stray print corrupts the JSON-RPC stream. Everything goes to stderr.
        log_level=settings_log_level(),
    )

    register_tools(server)
    register_resources(server)
    register_prompts(server)

    logger.info(
        "MCP server built: name=%s version=%s", SERVER_NAME, SERVER_VERSION
    )
    return server


def settings_log_level() -> str:
    from app.core.config import settings

    return settings.log_level.upper()


def build_asgi_app(stateless: bool = False):
    """
    Wrap the MCP server as a Starlette app for the Streamable HTTP transport.

    Exposed as an app factory (rather than only a `run()` helper) for two
    reasons:

    1. **Event loop control.** `server.run()` delegates to `anyio.run()`, which
       creates a ProactorEventLoop on Windows - the loop psycopg's async driver
       rejects. Serving the ASGI app with our own uvicorn call lets us pass a
       `SelectorEventLoop`, exactly like `app/run.py` does for the REST API.
    2. **Deployment.** Anything ASGI (uvicorn gunicorn workers, Docker, an
       existing gateway) can mount it without going through this CLI.

    Args:
        stateless: `True` serves the stateless `2026-07-28` dialect, where each
            request carries its own protocol version and capabilities in `_meta`
            and no session exists. That is the mode that scales with a plain
            round-robin load balancer. `False` serves the handshake session
            model, which is what most current MCP hosts speak.
    """
    server = build_server()
    app = server.streamable_http_app(
        # `json_response=True` returns plain JSON instead of an SSE stream.
        # Simpler for curl and for clients without SSE support. Switch to
        # `False` if you need server-initiated streaming notifications.
        json_response=True,
        stateless_http=stateless,
    )
    app.state.mcp_server = server
    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    # Allow `python -m app.mcp` from the backend directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(
        prog="python -m app.mcp",
        description="Run the Resume RAG MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio for local MCP hosts; http for remote/shared deployment",
    )
    parser.add_argument("--host", default="127.0.0.1", help="http transport only")
    parser.add_argument("--port", type=int, default=9000, help="http transport only")
    parser.add_argument(
        "--stateless",
        action="store_true",
        help=(
            "http only: serve the stateless 2026-07-28 dialect (no session, "
            "capabilities in _meta, scales with round-robin). Off by default "
            "because most current MCP hosts still speak the handshake session."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override LOG_LEVEL from .env (DEBUG/INFO/WARNING/ERROR)",
    )
    args = parser.parse_args(argv)

    # CRITICAL for stdio: stdout is the JSON-RPC transport, so logs MUST go to
    # stderr. This has to happen before anything logs - including the config
    # import above, which calls setup_logging() with the stdout default.
    setup_logging(stream=sys.stderr if args.transport == "stdio" else None)

    from app.core.runtime import configure_event_loop

    # Windows needs a SelectorEventLoop for async psycopg, exactly as in the
    # REST app. The MCP stdio runner creates its own loop, so this has to run
    # before it does.
    configure_event_loop()

    if args.transport == "stdio":
        server = build_server()
        # stdout is the protocol channel - log to stderr only.
        logger.info("starting MCP server on stdio")
        server.run(transport="stdio")
        return 0

    # HTTP: serve the ASGI app ourselves so the event loop is under our control.
    import uvicorn

    from app.core.runtime import selector_loop_factory

    app = build_asgi_app(stateless=args.stateless)
    dialect = "stateless 2026-07-28" if args.stateless else "handshake session"
    logger.info(
        "starting MCP server on http://%s:%s/mcp (%s)", args.host, args.port, dialect
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=(args.log_level or settings_log_level()).lower(),
        loop=selector_loop_factory() or "auto",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())