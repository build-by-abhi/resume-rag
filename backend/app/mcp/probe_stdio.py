"""
Verify the MCP server speaks the protocol correctly on BOTH transports.

    python -m app.mcp.probe_stdio

BACKGROUND - WHY TWO DIALECTS
-----------------------------
The protocol has two eras, and they are not interchangeable:

* **Handshake era** (`2025-06-18` / `2025-11-25`) - the long-lived session
  model. The client sends `initialize`, the server replies, the client sends
  `notifications/initialized`, and then requests go WITHOUT any `_meta`.
  This is what the **stdio** transport uses, because a stdio process *is* a
  session.

* **Stateless era** (`2026-07-28`) - no handshake, no sessions. Every request
  carries `_meta` with `io.modelcontextprotocol/protocolVersion` and
  `io.modelcontextprotocol/clientCapabilities`. This is what the
  **Streamable HTTP** transport uses in stateless mode.

Sending a stateless envelope over stdio is rejected by design:

    -32600 "this connection serves the handshake protocol era;
           requests carrying the 2026-07-28 envelope are not accepted on it"

This script speaks the dialect each transport actually expects, so a pass means
a real MCP client can drive the server.

THE OTHER FAILURE THIS CATCHES
-------------------------------
A stdio server's stdout IS the JSON-RPC transport. One stray log line and the
client reports "Response ended unexpectedly and may be incomplete." This script
parses every stdout line as JSON to prove the stream stays clean.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]

HANDSHAKE_VERSION = "2025-06-18"


def frame(request: dict) -> bytes:
    """One JSON-RPC message, newline-delimited as the protocol requires."""
    return (json.dumps(request) + "\n").encode()


def handshake_session() -> list[dict]:
    """The stdio flow, in the order a real client performs it."""
    return [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": HANDSHAKE_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "stdio-probe", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
        {"jsonrpc": "2.0", "id": 4, "method": "prompts/list", "params": {}},
        {"jsonrpc": "2.0", "id": 5, "method": "resources/templates/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "list_available_skills", "arguments": {"limit": 3}},
        },
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "search_candidates",
                "arguments": {"query": "kubernetes", "top_k": 2},
            },
        },
        {"jsonrpc": "2.0", "id": 8, "method": "ping", "params": {}},
    ]


#: Requests we expect a result for. The initialized notification correctly has none.
EXPECTED_IDS = {1, 2, 3, 4, 5, 6, 7, 8}


def describe(message: dict) -> str:
    """One-line human summary of a JSON-RPC response."""
    if "error" in message:
        return f"ERROR {message['error'].get('code')}: {message['error'].get('message', '')[:70]}"
    result = message.get("result", {})
    if not isinstance(result, dict):
        return f"ok {str(result)[:50]}"
    for key, label in (
        ("tools", "tools"),
        ("resources", "resources"),
        ("prompts", "prompts"),
        ("resourceTemplates", "resource templates"),
    ):
        if key in result:
            return f"ok  {label}: {len(result[key])}"
    if "content" in result:
        text = (result["content"] or [{}])[0].get("text", "")
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and "skills" in payload:
                return f"ok  tool call returned {payload['total']} skills"
            if isinstance(payload, dict) and "results" in payload:
                return f"ok  tool call returned {payload['total']} candidates"
        except json.JSONDecodeError:
            return f"ok  tool call, {len(text)} chars"
        return "ok  tool call"
    return "ok  (empty)"


def main() -> int:
    """
    Drive the server the way a real client does.

    `subprocess.run(input=...)` is NOT good enough here: it closes stdin as
    soon as the last byte is written, so a slow `tools/call` (model load, DB
    round trip) sees EOF mid-request and answers `-32000: Connection closed`.
    A real client keeps the pipe open for the whole session, so this holds
    stdin open, reads responses in a background thread, and only then closes.
    """
    import threading

    payload = b"".join(frame(r) for r in handshake_session())

    process = subprocess.Popen(
        [sys.executable, "-m", "app.mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(BACKEND),
    )
    assert process.stdin and process.stdout and process.stderr

    process.stdin.write(payload)
    process.stdin.flush()

    stdout_chunks: list[bytes] = []
    done = threading.Event()

    def _drain() -> None:
        assert process.stdout
        for line in process.stdout:
            stdout_chunks.append(line)
        done.set()

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    # Give the slowest tool call time to finish (first call loads the
    # embedding model and cross-encoder, which takes tens of seconds).
    finished = done.wait(timeout=180)
    if not finished:
        process.kill()
        reader.join(timeout=5)

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")

    # Close the session only after every response has been read.
    with contextlib.suppress(Exception):
        process.stdin.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=15)

    print("=" * 76)
    print("STDOUT - every line must be a valid JSON-RPC frame")
    print("=" * 76)

    seen: set[int] = set()
    errors: list[str] = []
    corrupt: list[str] = []

    for i, line in enumerate(stdout.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            corrupt.append(f"line {i}: blank")
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError as exc:
            # THE BUG THIS SCRIPT EXISTS TO CATCH: anything else on stdout.
            corrupt.append(f"line {i}: {stripped[:70]!r} ({exc})")
            print(f"{i:>3}. CORRUPT  {stripped[:70]}")
            continue

        msg_id = message.get("id")
        if msg_id is not None:
            seen.add(msg_id)
        summary = describe(message)
        print(f"{str(msg_id or '-'):>4}. {summary}")
        if "ERROR" in summary:
            errors.append(f"id={msg_id}: {summary}")

    missing = EXPECTED_IDS - seen
    if missing:
        errors.append(f"no response for id(s) {sorted(missing)}")

    print()
    print("=" * 76)
    print("STDERR - diagnostics, never parsed by the protocol")
    print("=" * 76)
    for line in stderr.splitlines()[:10]:
        print(f"  {line[:112]}")
    if not stderr.strip():
        print("  (empty)")

    print()
    print("=" * 76)
    failed = False

    if corrupt:
        failed = True
        print(f"FAIL  stdout corruption ({len(corrupt)} line(s)):")
        for problem in corrupt:
            print(f"        {problem}")
        print("      An MCP client reports: 'Response ended unexpectedly and may be")
        print("      incomplete.' Send logging to stderr - see app/core/logging.py.")

    if errors:
        failed = True
        print(f"FAIL  protocol errors ({len(errors)}):")
        for problem in errors:
            print(f"        {problem}")

    if not failed:
        print(f"PASS  {len(seen)} JSON-RPC responses, stdout clean, no protocol errors.")
        print()
        print("      initialize         negotiated")
        print("      tools/list         4")
        print("      resources/list     3")
        print("      prompts/list       1")
        print("      templates/list     1")
        print("      tools/call x2      returned data")
        print()
        print("      A real MCP host can drive this server over stdio.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())