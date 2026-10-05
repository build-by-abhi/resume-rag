"""
Inspect what the MCP server exposes, without starting a transport.

    python -m app.mcp.inspect

Useful when tuning tool descriptions or checking that a decorator change
actually registered. `MCPServer` exposes `list_tools` / `call_tool` /
`read_resource` directly, so the whole surface can be exercised in-process.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.logging import setup_logging  # noqa: E402

setup_logging()


async def show_surface() -> None:
    from app.mcp.server import build_server

    server = build_server()

    tools = await server.list_tools()
    print(f"\nTOOLS ({len(tools)})")
    print("=" * 78)
    for tool in tools:
        ann = tool.annotations
        flags = []
        if ann is not None:
            if getattr(ann, "read_only_hint", None):
                flags.append("read-only")
            else:
                flags.append("WRITES")
        print(f"\n  {tool.name}")
        print(f"    title       {tool.title}")
        print(f"    annotations {', '.join(flags) or 'none'}")
        print(f"    description {len(tool.description or '')} chars")
        # NOTE: mcp SDK 2.x exposes snake_case Python attributes
        # (`input_schema`), which serialise to camelCase on the wire
        # (`inputSchema`). Access the Python names, not the JSON ones.
        schema = tool.input_schema or {}
        required = schema.get("required") or []
        props = list(schema.get("properties") or {})
        print(f"    inputs      {', '.join(props) or 'none'}")
        print(f"    required    {required or 'none'}")
        print(f"    output      {'yes' if tool.output_schema else 'no'}")
        # First description line is what the model reads to decide on a call.
        first = (tool.description or "").strip().splitlines()[0]
        print(f"    hook        {first[:90]}")

    resources = await server.list_resources()
    print(f"\n\nRESOURCES ({len(resources)})")
    print("=" * 78)
    for res in resources:
        print(f"  {res.uri}")
        print(f"      {res.name} - {(res.description or '')[:70]}")

    templates = await server.list_resource_templates()
    if templates:
        print(f"\n\nRESOURCE TEMPLATES ({len(templates)})")
        print("=" * 78)
        for tpl in templates:
            print(f"  {tpl.uri_template}  ->  {tpl.name}")

    prompts = await server.list_prompts()
    print(f"\n\nPROMPTS ({len(prompts)})")
    print("=" * 78)
    for prompt in prompts:
        args = [a.name for a in (prompt.arguments or [])]
        print(f"  /{prompt.name}")
        print(f"      {(prompt.description or '')[:70]}")
        print(f"      args: {args}")


if __name__ == "__main__":
    from app.core.runtime import configure_event_loop, run_async

    configure_event_loop()
    raise SystemExit(run_async(show_surface()))