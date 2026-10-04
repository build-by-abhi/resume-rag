"""
Deterministic, dependency-free embedding provider.

WHAT IT IS
----------
A hashed bag-of-words projected into `dim` dimensions ("hashing trick" / random
indexing). Vocabulary is hashed into dimension indices and the counts are
weighted with sublinear TF and IDF-ish normalisation.

WHY IT EXISTS
-------------
It is NOT a semantic model and it will never be shipped to production. It lets
you:
  * run the full pipeline and the whole test suite with zero API keys,
  * have CI stay free and fast,
  * still get *lexically* meaningful similarity, so the hybrid search plumbing
    (filtering, RRF fusion, reranking, citations) is genuinely exercised.

For real semantic search set EMBEDDING_PROVIDER=sentence-transformers (free,
local) or openai (best quality).
"""

from __future__ import annotations

import hashlib
import math
import re

from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9+#.]+")


class HashingEmbeddingProvider(EmbeddingProvider):
    """Hashing-trick embeddings. Deterministic, offline, ~0ms per batch."""

    def __init__(self, dim: int = 1536) -> None:
        if dim < 64:
            raise ValueError("dim must be >= 64 for the hashing trick to be stable")
        self.dim = dim

    # -- API ----------------------------------------------------------------

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in self._prepared(texts)]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed((text or "").strip() or " ")

    # -- internals ----------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _WORD.findall(text.lower())

        for token, tf in self._term_counts(tokens).items():
            # Sublinear TF: "React" 8x should not be 8x as important as once.
            weight = (1.0 + math.log(tf)) * self._idf(token)
            idx, sign = self._bucket(token)
            vector[idx] += sign * weight

        # Add a light character 4-gram signal so near-miss spellings
        # ("pythonn") still land close to "python".
        for gram in self._char_grams(text.lower()):
            idx, sign = self._bucket("~" + gram)
            vector[idx] += sign * 0.15

        return self.l2_normalize(vector)

    @staticmethod
    def _term_counts(tokens: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    @staticmethod
    def _idf(token: str) -> float:
        """
        Static, corpus-free IDF approximation.

        We cannot compute real IDF without the corpus, so we approximate rarity by
        token length: short tokens ("js", "ai", "go") are rarer in resumes than
        long ones ("microservices"). This is a heuristic, not statistics.
        """
        return min(1.0, 0.45 + 0.055 * len(token))

    @staticmethod
    def _char_grams(text: str, n: int = 4) -> set[str]:
        compact = re.sub(r"\s+", " ", text.strip())
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[i : i + n] for i in range(len(compact) - n + 1)}

    def _bucket(self, token: str) -> tuple[int, float]:
        """
        Map a token to (dimension index, +/-1 sign).

        The signed hash is the standard "hashing with sign" trick: collisions
        cancel out instead of accumulating, so collisions add noise rather than
        false similarity.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, (1.0 if value >> 63 else -1.0)