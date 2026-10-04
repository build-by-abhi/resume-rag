"""
The embedding interface every backend must implement.

Why an interface instead of calling OpenAI inline?
- You can swap providers by changing one env var.
- Tests run offline with the deterministic `hashing` provider.
- The rest of the app depends on this contract, never on a vendor SDK.
"""

from __future__ import annotations

import abc
import math


class EmbeddingProvider(abc.ABC):
    """Turns text into fixed-length float vectors."""

    #: Dimensionality of the produced vectors. Must match the DB column width.
    dim: int

    @abc.abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (one resume chunk each)."""

    @abc.abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed one search query. Usually uses the same model as documents."""

    # -- shared helpers -----------------------------------------------------

    @staticmethod
    def l2_normalize(vector: list[float]) -> list[float]:
        """
        Scale a vector to length 1.

        Cosine distance = 1 - dot(a, b) only when both are unit length, so this
        keeps the SQL simple (`embedding <=> query_embedding`) and lets us
        compare similarity to a raw cosine score.
        """
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def _prepared(self, texts: list[str], max_chars: int = 8000) -> list[str]:
        """
        Providers reject empty/oversized input. Truncating here keeps every
        provider implementation trivial.
        """
        return [(t or " ").strip()[:max_chars] or " " for t in texts]