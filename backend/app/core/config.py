"""
Central application settings.

Everything is read from environment variables (typically a `.env` file), so the
same code runs locally, in Docker, and in CI without code changes.

`@lru_cache` means `get_settings()` builds the object once per process and
every module that imports it shares the same instance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# app/core/config.py  ->  app/core/../../  ->  backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Typed, validated view over the environment."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env", PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of crashing
    )

    # ---------------- App ----------------
    app_name: str = "Resume RAG API"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    # Comma-separated rather than a real list: pydantic-settings expects JSON
    # (`["a","b"]`) when a field is a list, and a comma-separated string is far
    # easier to write in a .env file. `cors_origin_list` below parses it.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---------------- Database ----------------
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "resume_rag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    # Optional explicit override; otherwise built from the parts above.
    database_url: str | None = None

    # ---------------- Embeddings ----------------
    # openai | sentence-transformers | huggingface | hashing
    embedding_provider: str = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 64

    # ---------------- LLM ----------------
    # openai | huggingface | anthropic | gemini | none
    llm_provider: str = "none"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    # One token works for the LLM and for embeddings on HuggingFace.
    huggingface_api_key: str | None = None
    # Extra JSON forwarded verbatim to the provider's chat endpoint.
    # Needed for hybrid reasoning models - see `llm_extra_body_merged`.
    llm_extra_body: str | None = None

    # ---------------- Ingestion ----------------
    max_upload_mb: int = 10
    chunk_size: int = 350
    chunk_overlap: int = 60
    upload_dir: Path = BACKEND_DIR / "storage" / "resumes"

    # ---------------- Search ----------------
    top_k: int = 10
    candidate_multiplier: int = 4
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_extra_body_merged(self) -> dict:
        """
        `LLM_EXTRA_BODY` merged over the provider's model-specific defaults.

        Why this exists
        ---------------
        `Qwen/Qwen3-*` is a *hybrid reasoning* model: it emits a `<think>...</think>`
        block before the answer unless told otherwise. For this application that
        is actively harmful:

          * it triples latency (irrelevant for a 6-field extraction)
          * the think block surrounds the JSON, so `parse_json_blob` must dig it
            out or the extraction silently falls back to rules
          * it would be rendered verbatim inside the answer panel

        The chat template honours `enable_thinking`, which is why it is passed
        through `extra_body` rather than being prompted away. Setting your own
        `LLM_EXTRA_BODY` always wins.
        """
        import json

        extra: dict = {}
        if self.llm_extra_body:
            try:
                parsed = json.loads(self.llm_extra_body)
                if isinstance(parsed, dict):
                    extra = parsed
            except json.JSONDecodeError:
                import logging

                logging.getLogger(__name__).warning(
                    "LLM_EXTRA_BODY is not valid JSON; ignoring it. "
                    'Example: LLM_EXTRA_BODY={"enable_thinking": false}'
                )

        model = (self.llm_model or "").lower()
        if model.startswith("qwen/qwen3") and "enable_thinking" not in extra:
            extra["enable_thinking"] = False

        return extra

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        """`CORS_ORIGINS` split into the list the middleware actually wants."""
        return [origin.strip() for origin in (self.cors_origins or "").split(",") if origin.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        """The URL SQLAlchemy actually connects with (async driver)."""
        if self.database_url:
            # Force the async driver even if someone wrote a sync URL in .env.
            return _to_async_url(self.database_url)
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _to_async_url(url: str) -> str:
    """Swap a sync postgres scheme for the async psycopg one."""
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql+"):
        # e.g. postgresql+asyncpg:// -> postgresql+psycopg://
        return url.split("+", 1)[0] + "+psycopg://" + url.split("://", 1)[1]
    return url


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Import this instead of constructing Settings yourself."""
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()