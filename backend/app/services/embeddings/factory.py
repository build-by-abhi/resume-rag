"""
Provider factory. One place that decides which embedding backend is active.

    from app.services.embeddings import get_embedder
    embedder = get_embedder()

The instance is cached, because loading a model is expensive.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.hashing_provider import HashingEmbeddingProvider

logger = get_logger(__name__)


@lru_cache
def get_embedder(provider: str | None = None) -> EmbeddingProvider:
    name = (provider or settings.embedding_provider).strip().lower()

    if name == "hashing":
        return HashingEmbeddingProvider(dim=settings.embedding_dim)

    if name == "openai":
        from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider()

    if name in {"sentence-transformers", "sentence_transformers", "st", "local"}:
        from app.services.embeddings.st_provider import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider()

    if name in {"huggingface", "hf"}:
        from app.services.embeddings.hf_provider import HuggingFaceEmbeddingProvider

        return HuggingFaceEmbeddingProvider()

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{name}'. "
        "Choose one of: hashing, openai, sentence-transformers, huggingface"
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper so callers don't need the provider instance."""
    if not texts:
        return []
    return await get_embedder().embed_documents(texts)


async def embed_query(text: str) -> list[float]:
    return await get_embedder().embed_query(text)