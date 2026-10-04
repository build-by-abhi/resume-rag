"""
Cross-encoder reranking (Phase 3, optional).

WHY RERANKING IS NEEDED
-----------------------
Bi-encoder embeddings (what we store) encode query and document *independently*.
That is fast -- which is why we can pre-compute document vectors -- but it means
the document vector was never aware of this specific query. A cross-encoder reads
the (query, chunk) pair TOGETHER, so it can catch things embeddings blur:
"5+ years of Kubernetes" vs "3 years of Kubernetes".

Cost: it is a forward pass per pair, so we only ever rerank the top ~50
candidates, never the whole corpus.

If `sentence-transformers` is not installed, `rerank` transparently returns the
input order -- retrieval degrades, it does not break.
"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model = None
_model_failed = False


class Reranker:
    """Lazy singleton wrapper around a CrossEncoder."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.rerank_model

    def _load(self):
        global _model, _model_failed
        if _model is None and not _model_failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("loading cross-encoder %s", self.model_name)
                _model = CrossEncoder(self.model_name, device="cpu")
            except Exception as exc:
                _model_failed = True
                logger.warning(
                    "reranker unavailable (%s). Install sentence-transformers or "
                    "set RERANK_ENABLED=false. Retrieval will skip reranking.",
                    exc,
                )
        return _model

    @property
    def available(self) -> bool:
        return self._load() is not None

    async def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        """
        Return relevance scores aligned with `documents`, or None if unavailable.
        Higher score = better match.
        """
        model = self._load()
        if model is None or not documents:
            return None

        pairs = [(query, doc) for doc in documents]

        def _predict() -> list[float]:
            scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
            return [float(s) for s in scores]

        # CPU-bound -> off the event loop.
        return await asyncio.to_thread(_predict)


reranker = Reranker()


async def rerank_chunks(
    query: str, documents: list[str], *, enabled: bool | None = None
) -> list[float] | None:
    """Config-aware helper. Returns None when reranking should be skipped."""
    use = settings.rerank_enabled if enabled is None else enabled
    if not use:
        return None
    return await reranker.rerank(query, documents)


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max scale to 0..1 so rerank scores can sit on the same axis as cosine
    similarity. A flat list maps to all-ones rather than dividing by zero.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]