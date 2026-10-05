"""
MCP (Model Context Protocol) server for the Resume RAG application.

WHAT THIS IS
------------
The REST API in `app/api/routes/` serves *your* frontend. This module serves an
LLM. Same retrieval engine, same database, different consumer - so an MCP host
(an IDE, Claude Desktop, another agent) can query the candidate pool without you
writing a second integration per client.

WHY THIS IS NOT A NEW CODE PATH
-------------------------------
Every handler here delegates to the existing services:

    MCP tool  ->  HybridSearcher / PgVectorStore / AnswerGenerator / RuleExtractor

That matters. The alternative - reimplementing search inside the MCP layer - is
how MCP servers end up quietly diverging from the real behaviour of the system
they claim to expose. A bug fixed in `hybrid.py` fixes the MCP surface for free.

THE PRIMITIVES AND WHO CONTROLS THEM
------------------------------------
    tools     the MODEL decides to call them   (search, match)
    resources the APP decides what to attach    (a candidate profile)
    prompts   the USER invokes them             (/screen_against_jd)

Choosing wrongly is the most common design error: putting a read-only profile
behind a tool means the model may never call it, and putting a mutating
operation behind a resource means it can fire without the model realising it
did something.

TRANSPORTS
----------
    stdio              local tool, spawned by the client (no auth needed)
    streamable-http    remote / shared / multi-tenant (OAuth framework)
"""

from __future__ import annotations

from app.mcp.server import build_server, main

__all__ = ["build_server", "main"]