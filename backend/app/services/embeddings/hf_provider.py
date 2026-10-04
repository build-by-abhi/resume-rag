"""
Embeddings via the HuggingFace Inference API.

Uses the same `HUGGINGFACE_API_KEY` as the LLM provider, so switching the whole
model stack to HuggingFace is a change of two env vars.

WHEN THIS IS WORTH IT OVER `sentence-transformers`
--------------------------------------------------
* `sentence-transformers` is free, offline, and faster on CPU. For this project
  it is the better default.
* This provider is useful when you want (a) no local model download, or (b) a
  stronger hosted embedding model such as `BAAI/bge-large-en-v1.5` without
  running it yourself.

WHAT IT COSTS
-------------
Each call is a network round trip. There is no free local compute, and the
free Inference tier is rate limited - so a 400-chunk bulk upload becomes 400
HTTP requests. Batch with `EMBEDDING_BATCH_SIZE` and expect slower ingestion.
For bulk ingestion, keep the local provider and switch to this only for search.
"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

#: Widths of commonly hosted embedding models. Must match EMBEDDING_DIM.
_MODEL_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "intfloat/e5-small-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
}


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str | None = None, dim: int | None = None) -> None:
        self.model = model or (
            "BAAI/bge-small-en-v1.5"
            # EMBEDDING_MODEL is often set for a different provider; do not send
            # a chat model id to the feature-extraction pipeline by accident.
            if settings.embedding_model.startswith("text-embedding")
            else settings.embedding_model
        )
        self.dim = dim or _MODEL_DIMS.get(self.model, settings.embedding_dim)
        self._client = None

    def _get_client(self):
        if self._client is None:
            key = settings.huggingface_api_key
            if not key:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=huggingface requires HUGGINGFACE_API_KEY in .env"
                )
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self.model, token=key)
        return self._client

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(self._prepared(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([(text or "").strip() or " "]))[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()

        def _call() -> list[list[float]]:
            # `feature_extraction` accepts a list and returns one row per input,
            # already normalised (unit length) so cosine similarity is a plain
            # dot product - which matches what the vector column expects.
            rows = client.feature_extraction(
                texts, normalize_embeddings=True, batch_size=settings.embedding_batch_size
            )
            return [[float(x) for x in row] for row in rows]

        try:
            vectors = await asyncio.to_thread(_call)
        except Exception as exc:
            raise RuntimeError(
                f"HuggingFace embedding call failed for {self.model}: {exc}"
            ) from exc

        if vectors and len(vectors[0]) != self.dim:
            raise RuntimeError(
                f"Model {self.model} returned {len(vectors[0])}-dim vectors but "
                f"EMBEDDING_DIM is {self.dim}. Fix .env, then re-create the vector "
                "columns: python -m scripts.seed --drop"
            )

        logger.debug("embedded %d texts via %s", len(texts), self.model)
        return [self.l2_normalize(v) for v in vectors]