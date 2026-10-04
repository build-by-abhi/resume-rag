"""
Local embeddings via `sentence-transformers`.

Best default for a portfolio project: free, no API key, no data leaving the
machine, and genuinely semantic. First call downloads the model (~90 MB) and
then it runs on CPU.

Install:  pip install sentence-transformers
Set:     EMBEDDING_PROVIDER=sentence-transformers
Note:    384 dimensions for all-MiniLM-L6-v2 -> set EMBEDDING_DIM=384 (and
         recreate the tables) before ingesting anything.
"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "intfloat/e5-small-v2": 384,
}


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str | None = None, dim: int | None = None) -> None:
        self.model_name = model or (
            "all-MiniLM-L6-v2"
            if settings.embedding_model.startswith("text-embedding")
            else settings.embedding_model
        )
        self.dim = dim or _MODEL_DIMS.get(self.model_name, settings.embedding_dim)
        self._model = None
        self._lock = asyncio.Lock()

    def _get_model(self):
        """Load lazily (and once) - downloading happens on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading sentence-transformers model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
            actual = self._model.get_sentence_embedding_dimension()
            if actual != self.dim:
                raise RuntimeError(
                    f"Model {self.model_name} produces {actual}-dim vectors but "
                    f"EMBEDDING_DIM is {self.dim}. Fix .env and recreate the tables."
                )
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(self._prepared(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([(text or "").strip() or " "]))[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with self._lock:
            model = self._get_model()
            # sentence-transformers is blocking/CPU-bound -> keep it off the
            # event loop so the API stays responsive during a bulk upload.
            vectors = await asyncio.to_thread(
                model.encode,
                texts,
                batch_size=settings.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        normalized = [self.l2_normalize([float(x) for x in row]) for row in vectors]
        logger.debug("embedded %d texts locally", len(texts))
        return normalized