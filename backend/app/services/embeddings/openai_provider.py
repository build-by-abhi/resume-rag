"""OpenAI embeddings (`text-embedding-3-small` by default). Best quality."""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

# Widths of the models we might be asked for. The DB column is created from
# EMBEDDING_DIM, so this must match .env.
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str | None = None, dim: int | None = None) -> None:
        self.model = model or settings.embedding_model
        self.dim = dim or _MODEL_DIMS.get(self.model, settings.embedding_dim)
        self._client = None

    def _get_client(self):
        """Lazily import + build the client so import time stays cheap."""
        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY in .env"
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(self._prepared(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([(text or "").strip() or " "]))[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        vectors: list[list[float]] = []

        # Batch because the API caps tokens per request.
        batch = settings.embedding_batch_size
        for start in range(0, len(texts), batch):
            window = texts[start : start + batch]
            response = await client.embeddings.create(
                model=self.model, input=window, dimensions=self.dim
            )
            # Sort by `index`: the API does not guarantee response order.
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda d: d.index)
            )

        logger.debug("embedded %d texts with %s", len(texts), self.model)
        return [self.l2_normalize(v) for v in vectors]