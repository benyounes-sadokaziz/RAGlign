"""Dense retrieval: exact cosine top-k over a numpy matrix (TC-4).

No vector database. At this scale the index is a few megabytes and brute force
is both exact and sub-millisecond; an ANN index would add approximate recall --
a confound in a study whose subject is retrieval correctness -- plus a
persistence layer that can go stale against the corpus.

Vectors are L2-normalised at embed time, so cosine similarity is `matrix @ q`.
"""

from __future__ import annotations

import numpy as np

from ..embedding import Embedder
from ..models import Chunk
from .base import Retriever, ScoredChunk


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.encode_chunks(chunks)  # [n_chunks, dim], normalised

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        # argpartition then sort only the top k: O(n) instead of O(n log n).
        # Irrelevant at 800 chunks, but it costs one line and removes a scaling
        # cliff if the corpus grows.
        k = min(k, len(self.chunks))
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in idx]
