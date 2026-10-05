"""
Probe the Streamable HTTP transport.

    python -m app.mcp.probe_http [base_url]

Covers the handshake-session dialect, which is the default for
`python -m app.mcp --transport http`:

    POST /mcp  initialize            -> 200 + `Mcp-Session-Id` response header
    POST /mcp  notifications/initialized   (with that header)
    POST /mcp  tools/list, tools/call      (with that header)

Run the server first:

    python -m app.mcp --transport http --port 9101

For the stateless `2026-07-28` dialect, start the server with `--stateless` and
use `python -m app.mcp.probe_http --stateless`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:9101/mcp"
HANDSHAKE_VERSION = "2025-06-18"
STATELESS_VERSION = "2026-07-28"


class Client:
    """Minimal MCP HTTP client: enough to prove the transport works."""

    def __init__(self, base: str, stateless: bool) -> None:
        self.base = base
        self.stateless = stateless
        self.session_id: str | None = None

    def _headers(self, method: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": STATELESS_VERSION if self.stateless else HANDSHAKE_VERSION,
            # Standard routing headers required on Streamable HTTP POSTs.
            "Mcp-Method": method,
            "Mcp-Name": method.rsplit("/", 1)[-1],
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def post(self, payload: dict) -> tuple[int, dict | str]:
        request = urllib.request.Request(
            self.base,
            data=json.dumps(payload).encode(),
            headers=self._headers(payload.get("method", "")),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.headers.get("Mcp-Session-Id"):
                    self.session_id = response.headers["Mcp-Session-Id"]
                body = response.read().decode("utf-8", "replace").strip()
                return response.status, _try_json(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace").strip()
            return exc.code, _try_json(body)

    def _meta(self) -> dict:
        if not self.stateless:
            return {}
        return {
            "io.modelcontextprotocol/protocolVersion": STATELESS_VERSION,
            "io.modelcontextprotocol/clientCapabilities": {},
        }


def _try_json(text: str):
    """Parse a JSON body, unwrapping the SSE `data:` frames if present."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                return json.loads(line.removeprefix("data:").strip())
            except json.JSONDecodeError:
                continue
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the MCP HTTP transport.")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Probe the stateless 2026-07-28 dialect (server started with --stateless)",
    )
    args = parser.parse_args()

    client = Client(args.base, args.stateless)
    dialect = "stateless 2026-07-28" if args.stateless else "handshake session"
    print("=" * 74)
    print(f"Streamable HTTP - {args.base}  ({dialect})")
    print("=" * 74)

    status, body = client.post(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": STATELESS_VERSION if args.stateless else HANDSHAKE_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "http-probe", "version": "1.0"},
            },
        }
    )
    print(f"  initialize          HTTP {status}")
    if isinstance(body, dict) and "error" in body:
        print(f"    ERROR {body['error']}")
        return 1
    caps = (body.get("result") or {}).get("capabilities", {}) if isinstance(body, dict) else {}
    print(f"    capabilities: {', '.join(sorted(caps)) or 'none'}")
    print(f"    session: {client.session_id or '(stateless - none issued)'}")

    if not args.stateless:
        client.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        print("  initialized         (notification, no response expected)")

    for method, req_id in (("tools/list", 2), ("resources/list", 3), ("prompts/list", 4)):
        status, body = client.post(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": {"_meta": client._meta()},
            }
        )
        if isinstance(body, dict) and "error" in body:
            print(f"  {method:<20} HTTP {status}  ERROR {body['error'].get('message')}")
            continue
        result = body.get("result", {}) if isinstance(body, dict) else {}
        counts = {k: len(result[k]) for k in ("tools", "resources", "prompts") if k in result}
        print(f"  {method:<20} HTTP {status}  {counts}")

    status, body = client.post(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "search_candidates",
                "arguments": {"query": "kubernetes", "top_k": 2},
                "_meta": client._meta(),
            },
        }
    )
    if isinstance(body, dict) and "error" in body:
        print(f"  tools/call          HTTP {status}  ERROR {body['error'].get('message')}")
        return 1
    result = body.get("result", {}) if isinstance(body, dict) else {}
    text = (result.get("content") or [{}])[0].get("text", "")
    try:
        payload = json.loads(text)
        print(f"  tools/call          HTTP {status}  {payload.get('total')} candidates")
        for row in payload.get("results", [])[:2]:
            print(f"      - {row.get('full_name')} ({row.get('current_title')})")
    except json.JSONDecodeError:
        print(f"  tools/call          HTTP {status}  (unparsed payload)")

    print()
    print("PASS" if status == 200 else "FAIL", "- the Streamable HTTP transport works")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())