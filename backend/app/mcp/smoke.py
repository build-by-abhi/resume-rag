"""
Exercise every MCP primitive against the live database, in-process.

    python -m app.mcp.smoke

`MCPServer.call_tool` / `read_resource` / `get_prompt` invoke handlers directly,
so this verifies real behaviour (SQL, retrieval, embeddings) without needing an
MCP client or a running transport.

This is the fastest way to confirm the server actually works after changing a
tool signature or a schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.logging import setup_logging  # noqa: E402

setup_logging()


def show(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def payload_of(result) -> dict:
    """
    Pull the structured payload out of a CallToolResult.

    MCP returns both `structuredContent` and a serialised TextContent block for
    backward compatibility. Prefer the structured form.
    """
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    return {}


async def main() -> int:
    from app.mcp.context import database
    from app.mcp.server import build_server

    server = build_server()

    # ---- resource: corpus stats --------------------------------------
    show("RESOURCE  talent://corpus/stats")
    async for contents in _read(server, "talent://corpus/stats"):
        print(json.dumps(json.loads(contents.content), indent=2))

    # ---- resource: skill facets --------------------------------------
    show("RESOURCE  talent://facets/skills  (top 8)")
    async for contents in _read(server, "talent://facets/skills"):
        data = json.loads(contents.content)
        for skill in data["skills"][:8]:
            print(f"  {skill['display']:<22} {skill['candidate_count']} candidate(s)")

    # ---- resource: directory -----------------------------------------
    show("RESOURCE  resume://candidates")
    async for contents in _read(server, "resume://candidates"):
        data = json.loads(contents.content)
        print(f"  total: {data['total']}")
        for c in data["candidates"][:5]:
            years = c["total_years_experience"]
            print(f"  - {str(c['full_name']):<24} {str(c['current_title'])[:26]:<28} {years} yrs")

    # ---- tool: skills -------------------------------------------------
    show("TOOL  list_available_skills(limit=5)")
    result = await server.call_tool("list_available_skills", {"limit": 5})
    print(json.dumps(payload_of(result), indent=2))

    # ---- tool: search -------------------------------------------------
    show("TOOL  search_candidates(query='event streaming pipelines at scale')")
    result = await server.call_tool(
        "search_candidates",
        {"query": "event streaming pipelines at scale", "top_k": 3},
    )
    data = payload_of(result)
    print(f"  total={data['total']}  diagnostics={data.get('diagnostics')}")
    for row in data["results"]:
        print(f"\n  {row['full_name']} - {row['current_title']}")
        print(f"    skills: {', '.join(row['skills'][:8])}")
        if row["evidence"]:
            ev = row["evidence"][0]
            print(f"    evidence[{ev['section']}]: {ev['content'][:110]}...")

    # ---- tool: filters only -------------------------------------------
    show("TOOL  search_candidates(skills=['kubernetes','terraform'], min_years_experience=5)")
    data = payload_of(
        await server.call_tool(
            "search_candidates",
            {"skills": ["kubernetes", "terraform"], "min_years_experience": 5, "top_k": 10},
        )
    )
    print(f"  total={data['total']}")
    for row in data["results"]:
        print(f"  - {row['full_name']:<24} {row['total_years_experience']} yrs  {row['location']}")

    # ---- tool: rejection path ------------------------------------------
    show("TOOL  search_candidates() with nothing to search on")
    print(json.dumps(payload_of(await server.call_tool("search_candidates", {})), indent=2))

    # ---- tool: JD match ------------------------------------------------
    show("TOOL  match_job_description")
    jd = (
        "Senior JVM backend engineer. Build event-driven services with Kotlin, "
        "Spring Boot and Kafka, own a PostgreSQL migration, run on Kubernetes."
    )
    data = payload_of(
        await server.call_tool(
            "match_job_description", {"job_description": jd, "top_k": 3}
        )
    )
    print(f"  total={data['total']}  scoring={data.get('scoring')}")
    for row in data["results"]:
        print(f"  - {str(row['full_name']):<24} {str(row['current_title'])[:30]}")

    # ---- tool: short JD is rejected ------------------------------------
    show("TOOL  match_job_description(too short) -> validation error")
    result = await server.call_tool("match_job_description", {"job_description": "hire"})
    print(json.dumps(payload_of(result), indent=2))

    # ---- resource template: one candidate ------------------------------
    show("RESOURCE TEMPLATE  resume://candidates/{id}")
    directory = json.loads((await _collect(server, "resume://candidates"))[0])
    if directory["candidates"]:
        cid = directory["candidates"][0]["id"]
        async for contents in _read(server, f"resume://candidates/{cid}"):
            data = json.loads(contents.content)
            print(f"  {data['full_name']} - {data['current_title']}")
            print(f"  skills: {len(data['skills'])}   chunks: {data['chunk_count']}")
            if data["chunks"]:
                c = data["chunks"][0]
                print(f"  chunk[0] ({c['section']}): {c['content'][:100]}...")

    show("RESOURCE  resume://candidates/does-not-exist -> graceful error")
    async for contents in _read(server, "resume://candidates/00000000-0000-0000-0000-000000000000"):
        print(contents.content)

    # ---- prompt ---------------------------------------------------------
    show("PROMPT  /screen_against_jd")
    result = await server.get_prompt(
        "screen_against_jd",
        {"job_description": "Senior Python engineer", "max_candidates": "3"},
    )
    text = result.messages[0].content.text
    print(f"  {len(text)} chars, first lines:")
    for line in text.splitlines()[:6]:
        print(f"    {line}")

    await database.aclose()
    print("\nAll MCP primitives responded.")
    return 0


async def _read(server, uri: str):
    """
    Await `read_resource` and normalise its return shape.

    The SDK returns `Iterable[ReadResourceContents]`, but depending on version
    that iterable is produced by an async generator - hence the await, then the
    check for both sync and async iteration.
    """
    result = await server.read_resource(uri)
    if hasattr(result, "__aiter__"):
        async for item in result:
            yield item
    else:
        for item in result:
            yield item


async def _collect(server, uri: str) -> list:
    out = []
    async for item in _read(server, uri):
        out.append(item.content)
    return out


if __name__ == "__main__":
    # run_async, not asyncio.run: the coroutine touches Postgres, and Windows'
    # default ProactorEventLoop is rejected by psycopg's async driver.
    from app.core.runtime import configure_event_loop, run_async

    configure_event_loop()
    raise SystemExit(run_async(main()))