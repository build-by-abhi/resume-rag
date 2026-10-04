"""
Check that the configured LLM provider actually works.

    python scripts/check_llm.py
    python scripts/check_llm.py --question "which candidates have Go experience?"

WHY THIS EXISTS
---------------
Provider problems are otherwise discovered as a silent quality regression: the
app logs "no API key -> extractive fallback" and keeps working, so you never
notice your key expired or your Llama licence was never accepted. This fails
loudly instead.

It checks three things in order of how often they are the problem:
  1. a key is configured at all
  2. the model is reachable and not gated (401/403/404/503 get specific advice)
  3. the model returns JSON when asked for JSON (needed by the extractor)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow `python scripts/check_llm.py` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.prompts.templates import FILTER_PARSE_SYSTEM, FILTER_PARSE_USER  # noqa: E402
from app.services.llm.client import (  # noqa: E402
    LLMError,
    NullLLMClient,
    build_llm_client,
)

logger = get_logger("check_llm")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the configured LLM provider.")
    parser.add_argument(
        "--question",
        default="senior Python engineer with Kubernetes experience in Berlin",
        help="Question used for the JSON round trip test",
    )
    args = parser.parse_args()

    setup_logging()

    print("=" * 68)
    print(f"  LLM_PROVIDER = {settings.llm_provider}")
    print(f"  LLM_MODEL    = {settings.llm_model}")
    print("=" * 68)

    if settings.llm_provider in {"huggingface", "hf"}:
        key = settings.huggingface_api_key
        if key:
            # Show only the shape, never the secret.
            print(f"  HUGGINGFACE_API_KEY = {key[:7]}...{key[-4:]} ({len(key)} chars)")
        else:
            print("  HUGGINGFACE_API_KEY = <not set>")

    client = build_llm_client()

    if isinstance(client, NullLLMClient):
        print("\n  RESULT: no LLM configured.\n")
        print("  The app still works - search, filters and JD matching all run, and the")
        print("  answer panel serves an extractive digest instead of generated text.\n")
        print("  To enable one:")
        print("    LLM_PROVIDER=huggingface")
        print("    LLM_MODEL=Qwen/Qwen3-8B      # not gated, plain read token")
        print("    HUGGINGFACE_API_KEY=hf_...")
        return 1

    print(f"\n  client = {client.name}  model = {client.model}")

    # --- 1. plain completion ------------------------------------------
    # A generous budget on purpose. Qwen3 (like several other chat models)
    # prefixes its answer with whitespace; with a tight max_tokens the provider
    # returns finish_reason="length" and empty content, which looks identical to
    # a dead key. 300 leaves room for that preamble.
    print("\n  [1/2] plain completion")
    try:
        response = await client.complete(
            "You are a helpful assistant. Answer in one short sentence.",
            "What is the capital of France?",
            max_tokens=300,
        )
    except LLMError as exc:
        print(f"  FAILED: {exc}\n")
        return 2

    if response.empty:
        print(
            f"  FAILED: the provider returned an empty response "
            f"(finish_reason={response.finish_reason})\n"
        )
        return 2

    print(f"  ok in {response.latency_ms:.0f}ms -> {response.text.strip()[:160]!r}")
    print(
        f"  tokens: prompt={response.prompt_tokens} completion={response.completion_tokens}"
        f"  finish_reason={response.finish_reason}"
    )
    if response.truncated:
        print("  note: response was truncated (raise max_tokens in the caller)")

    # --- 2. JSON mode (the extractor depends on this) ------------------
    print("\n  [2/2] JSON round trip (used by structured extraction)")
    payload = await client.complete_json(
        FILTER_PARSE_SYSTEM, FILTER_PARSE_USER.format(question=args.question), max_tokens=400
    )

    if payload is None:
        print("  WARNING: no parseable JSON returned.")
        print("  Extraction will fall back to rules only, and JD scoring will use the")
        print("  retrieval score. That is survivable, but it is a real quality loss.")
        print("  A larger model, or a few-shot example in the prompt, usually fixes it.")
        return 0

    print(f"  ok -> {payload}")
    print("\n  RESULT: provider healthy.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))