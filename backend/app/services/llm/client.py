"""
Provider-agnostic LLM client.

One interface (`LLMClient.complete` / `complete_json`) with three backends.
Every method degrades gracefully when no API key is configured, which keeps the
app runnable offline - the search endpoints still work, you just don't get a
written summary sentence.
"""

from __future__ import annotations

import abc
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Upper bound for the automatic retry when a response is truncated to nothing.
_MAX_RETRY_TOKENS = 4096


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    #: "stop" | "length" | ... - `length` means the token budget ran out.
    finish_reason: str | None = None

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMError(RuntimeError):
    """Raised when a provider call fails in a way we cannot recover from."""


class LLMClient(abc.ABC):
    """Minimal contract every provider implements."""

    name: str
    model: str

    @abc.abstractmethod
    async def complete(
        self, system: str, user: str, *, max_tokens: int = 700, temperature: float | None = None
    ) -> LLMResponse:
        """Single-turn completion."""

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int = 1200, temperature: float = 0.0
    ) -> dict | list | None:
        """
        Completion constrained to valid JSON.

        Providers disagree about JSON mode, so we ask nicely *and* parse
        defensively: strip code fences, grab the outermost object, then
        `json.loads`. Returns None when nothing usable came back.
        """
        json_system = (
            system.strip()
            + "\n\nRespond with a single valid JSON value and nothing else. "
            "No prose, no markdown fences, no trailing commas."
        )
        try:
            response = await self.complete(
                json_system, user, max_tokens=max_tokens, temperature=temperature
            )
        except LLMError:
            return None

        return parse_json_blob(response.text)


#: Reasoning traces emitted by hybrid thinking models (Qwen3, DeepSeek-R1, ...).
#: Formats seen in the wild: <think>...</think>, <reasoning>...</reasoning>,
#: and an unterminated opening tag when max_tokens cut the response short.
_THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_UNCLOSED_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """
    Remove chain-of-thought blocks from a model response.

    Two reasons this is not optional for `Qwen/Qwen3-*`:

    1. Extraction. The reasoning sits between the prompt and the JSON, so
       `parse_json_blob` has to survive it. We already dig out a `{...}` block,
       but a long trace makes that fragile - and if it fails, extraction quietly
       reverts to rules and the UI still looks fine.
    2. The answer panel. Raw reasoning rendered to a recruiter is noise, and it
       is model-internal content, not an answer.

    Returns the text unchanged when there is nothing to strip.
    """
    if not text or "<" not in text:
        return text or ""

    cleaned = _THINK_BLOCK.sub("", text)
    # max_tokens can cut the response mid-trace, leaving an unterminated tag.
    cleaned = _UNCLOSED_THINK_BLOCK.sub("", cleaned)
    return cleaned.strip()


def parse_json_blob(text: str) -> dict | list | None:
    """Best-effort extraction of a JSON value from an LLM response."""
    if not text:
        return None
    candidate = text.strip()

    # ```json ... ``` fences
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    parsed = _try_parse(candidate)
    if parsed is not None:
        return parsed

    # Trailing commas. Small models (3B-class and below) emit `{"a": 1,}` often
    # enough that ignoring it costs a whole extraction pass. Only attempted after
    # strict parsing failed.
    #
    # Caveat: this is a textual repair, so a string value that itself ends with
    # `, }` would be mangled. That is strictly better than discarding an
    # otherwise valid response, and it cannot affect any prompt in this repo.
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    if repaired != candidate:
        parsed = _try_parse(repaired)
        if parsed is not None:
            return parsed

    # Fall back to the outermost {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            parsed = _try_parse(candidate[start : end + 1])
            if parsed is not None:
                return parsed
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate[start : end + 1])
            parsed = _try_parse(repaired)
            if parsed is not None:
                return parsed

    logger.warning("could not parse JSON from LLM response")
    return None


def _try_parse(text: str) -> dict | list | None:
    """`json.loads` that distinguishes 'invalid' from 'parsed null/empty'."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class OpenAIClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        self.name = "openai"
        self.model = model or settings.llm_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not settings.openai_api_key:
                raise LLMError("OPENAI_API_KEY is not set")
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 700, temperature: float | None = None
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=(
                    settings.llm_temperature if temperature is None else temperature
                ),
                max_tokens=max_tokens,
            )
        except Exception as exc:  # network, auth, quota, bad model name...
            raise LLMError(f"openai call failed: {exc}") from exc

        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int = 1200, temperature: float = 0.0
    ) -> dict | list | None:
        json_system = (
            system.strip()
            + "\n\nRespond with a single valid JSON value and nothing else. "
            "No prose, no markdown fences, no trailing commas."
        )
        try:
            response = await self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": json_system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                # Native JSON mode: much more reliable than parsing prose.
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMError(f"openai call failed: {exc}") from exc

        return parse_json_blob(response.choices[0].message.content or "")


class AnthropicClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        self.name = "anthropic"
        self.model = model or settings.llm_model

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 700, temperature: float | None = None
    ) -> LLMResponse:
        started = time.perf_counter()
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=(
                    settings.llm_temperature if temperature is None else temperature
                ),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError(f"anthropic call failed: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=getattr(response.usage, "input_tokens", None),
            completion_tokens=getattr(response.usage, "output_tokens", None),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class GeminiClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        self.name = "gemini"
        self.model = model or settings.llm_model

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 700, temperature: float | None = None
    ) -> LLMResponse:
        started = time.perf_counter()
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        started = time.perf_counter()
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config={
                    "system_instruction": system,
                    "max_output_tokens": max_tokens,
                    "temperature": (
                        settings.llm_temperature if temperature is None else temperature
                    ),
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:
            raise LLMError(f"gemini call failed: {exc}") from exc

        return LLMResponse(
            text=response.text or "",
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class HuggingFaceClient(LLMClient):
    """
    HuggingFace Inference API, via the official `huggingface_hub` client.

    WHY THE OFFICIAL CLIENT RATHER THAN RAW HTTP
    ---------------------------------------------
    HuggingFace has moved the serverless API around several times
    (`api-inference.huggingface.co`, then the OpenAI-compatible
    `router.huggingface.co/hf-inference/.../v1/chat/completions`, plus per-provider
    routes). `InferenceClient` hides all of that, forwards your token, and
    retries across providers - so this provider does not break when HF changes
    an endpoint again.

    GATED MODELS
    ------------
    `meta-llama/*` is gated. Before this works the account must:
      1. open the model page on huggingface.co and accept the licence
      2. create a token with "Read access to contents of all public gated
         repos you can access"
    A valid key without the licence grant returns 401/403, not a model error -
    the message here says so explicitly because that is a confusing first run.
    """

    name = "huggingface"
    #: HF's serverless API is behind an OpenAI-compatible surface.
    supports_native_json_mode = False

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.llm_model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            key = self._api_key or settings.huggingface_api_key
            if not key:
                raise LLMError("HUGGINGFACE_API_KEY is not set")
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise LLMError(
                    "huggingface_hub is not installed. "
                    "Run: pip install 'huggingface_hub>=0.30'"
                ) from exc

            self._client = InferenceClient(model=self.model, token=key)
        return self._client

    @property
    def extra_body(self) -> dict:
        """
        Provider-specific request fields (e.g. Qwen3's `enable_thinking`).

        Read per call rather than cached so a test can change settings without
        rebuilding the client.
        """
        return getattr(settings, "llm_extra_body_merged", {}) or {}

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 700, temperature: float | None = None
    ) -> LLMResponse:
        import asyncio
        import time

        started = time.perf_counter()
        client = self._get_client()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        extra = self.extra_body
        budget = max_tokens

        def _call(budget_tokens: int) -> Any:
            # `chat_completion` is synchronous (network-bound), so it must not
            # run on the event loop.
            kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": budget_tokens,
                "temperature": (
                    settings.llm_temperature if temperature is None else temperature
                ),
            }
            if extra:
                # Only send `extra_body` when there is something to send:
                # some providers reject unknown fields outright.
                kwargs["extra_body"] = extra
            return client.chat_completion(**kwargs)

        prompt_tokens = 0
        completion_tokens = 0
        for attempt in range(2):
            try:
                raw = await asyncio.to_thread(_call, budget)
            except Exception as exc:
                raise LLMError(_explain_hf_error(exc, self.model)) from exc

            choices = getattr(raw, "choices", None) or []
            content = ""
            if choices:
                content = getattr(choices[0].message, "content", None) or ""

            finish_reason = getattr(choices[0], "finish_reason", None) if choices else None

            usage = getattr(raw, "usage", None)
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            # Models frequently emit leading whitespace or a preamble before the
            # real content. With a tight budget that preamble can consume the
            # whole allowance, and the provider then returns
            # finish_reason="length" with content="" - which looks exactly like a
            # broken provider. Retry once with more room rather than surfacing
            # an empty answer to the user.
            if finish_reason == "length" and not content.strip() and attempt == 0:
                larger = min(budget * 4, _MAX_RETRY_TOKENS)
                logger.warning(
                    "%s returned an empty response (finish_reason=length) with "
                    "max_tokens=%d; retrying with %d",
                    self.model,
                    budget,
                    larger,
                )
                budget = larger
                continue

            return LLMResponse(
                # `.lstrip()` because Qwen3 prefixes answers with "\n\n".
                text=strip_reasoning(content).lstrip(),
                model=self.model,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens or None,
                latency_ms=(time.perf_counter() - started) * 1000,
                finish_reason=finish_reason,
            )

        # Unreachable: the loop always returns on the final attempt.
        raise LLMError(f"{self.model} produced no response")

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int = 1200, temperature: float = 0.0
    ) -> dict | list | None:
        """
        Ask for JSON in the prompt and parse defensively.

        The base-class implementation is exactly right here: `InferenceClient`
        has no reliable cross-model equivalent of OpenAI's `response_format`, so
        we request JSON in the system prompt and then rely on `parse_json_blob`
        to survive code fences and trailing prose.
        """
        json_system = (
            system.strip()
            + "\n\nRespond with a single valid JSON value and nothing else. "
            "No prose, no markdown fences, no trailing commas."
        )
        try:
            response = await self.complete(
                json_system, user, max_tokens=max_tokens, temperature=temperature
            )
        except LLMError:
            return None
        return parse_json_blob(response.text)


def _explain_hf_error(exc: Exception, model: str) -> str:
    """
    Turn a HuggingFace HTTP failure into something actionable.

    The two failure modes people actually hit are 401/403 (token or licence) and
    503 (provider overloaded / model not served), and the raw messages for both
    are unhelpful.
    """
    message = str(exc)
    lowered = message.lower()

    if "401" in message or "unauthorized" in lowered or "invalid token" in lowered:
        return (
            f"HuggingFace rejected the token ({model}). Check HUGGINGFACE_API_KEY, "
            "and confirm the token has 'Read access to contents of all public "
            "gated repos'. Llama models are gated: you must also accept the "
            f"licence at huggingface.co/{model}"
        )
    if "403" in message or "forbidden" in lowered or "gated" in lowered:
        return (
            f"Access to {model} is gated and your token has not been granted it. "
            f"Accept the licence at https://huggingface.co/{model}, then create "
            "a new token with gated-repo read access."
        )
    if "503" in message or "overloaded" in lowered:
        return f"The HuggingFace provider for {model} is overloaded. Retry shortly."
    if "404" in message:
        return f"Model '{model}' was not found on HuggingFace Inference. Check the repo id."
    return f"huggingface call failed for {model}: {message[:300]}"


class NullLLMClient(LLMClient):
    """Used when LLM_PROVIDER=none or no key is configured."""

    name = "none"
    model = "none"

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 700, temperature: float | None = None
    ) -> LLMResponse:
        logger.info("LLM disabled (LLM_PROVIDER=none); returning empty response")
        return LLMResponse(text="", model="none")

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int = 1200, temperature: float = 0.0
    ) -> dict | list | None:
        return None


def build_llm_client(provider: str | None = None) -> LLMClient:
    """
    Pick a provider. Falls back to `NullLLMClient` instead of crashing, because
    search must work without an API key - only the natural-language answer is lost.
    """
    name = (provider or settings.llm_provider or "none").strip().lower()

    if name == "openai" and settings.openai_api_key:
        return OpenAIClient()
    if name in {"huggingface", "hf"} and settings.huggingface_api_key:
        return HuggingFaceClient()
    if name in {"anthropic", "claude"} and settings.anthropic_api_key:
        return AnthropicClient()
    if name in {"gemini", "google"} and settings.gemini_api_key:
        return GeminiClient()

    if name not in {"none", ""}:
        reason = (
            "the key is not set"
            if name in {"huggingface", "hf"}
            else "no API key is configured"
        )
        logger.warning(
            "LLM provider '%s' requested but %s; falling back to no-LLM mode "
            "(search and filters still work, the answer is extractive).",
            name,
            reason,
        )
    return NullLLMClient()


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Cached singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = build_llm_client()
    return _llm_client


def llm_available() -> bool:
    return not isinstance(get_llm_client(), NullLLMClient)


#: Which env var holds the key for each provider. Used to tell the user
#: *which* setting is missing instead of a vague "no LLM key configured".
_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "hf": "HUGGINGFACE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}


def llm_status() -> dict:
    """
    Explain the current LLM configuration in a form the UI can display.

    The distinction that matters: "no provider selected" and "provider selected
    but the key is empty" are different mistakes, and reporting both as
    "no LLM key configured" sends people looking in the wrong file.
    """
    provider = (settings.llm_provider or "none").strip().lower()

    if provider in {"none", ""}:
        return {
            "configured": False,
            "provider": "none",
            "model": settings.llm_model,
            "reason": (
                "LLM_PROVIDER=none in .env - set it to "
                "huggingface/openai/anthropic/gemini to enable written answers."
            ),
        }

    env_var = _PROVIDER_KEY_ENV.get(provider)
    raw = getattr(settings, env_var.lower(), None) if env_var else None
    # A stray space in .env (`HUGGINGFACE_API_KEY= `) reads as "configured" but
    # produces a 401. Treat blank as missing.
    value = raw.strip() if isinstance(raw, str) else raw

    if not value:
        return {
            "configured": False,
            "provider": provider,
            "model": settings.llm_model,
            "reason": (
                f"LLM_PROVIDER={provider} is set but {env_var} is empty in backend/.env."
            ),
            "env_var": env_var,
        }

    if provider in {"huggingface", "hf"} and not _huggingface_hub_installed():
        return {
            "configured": False,
            "provider": provider,
            "model": settings.llm_model,
            "reason": (
                "huggingface_hub is not installed. "
                "Run: pip install 'huggingface_hub>=0.30'"
            ),
        }

    return {
        "configured": True,
        "provider": provider,
        "model": settings.llm_model,
        "reason": None,
    }


def _huggingface_hub_installed() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True