"""Hybrid retrieval by Reciprocal Rank Fusion.

RRF rather than a weighted sum of scores, for a concrete reason: BM25 scores are
unbounded and corpus-dependent while cosine similarities sit in [-1, 1], so any
weighted sum needs per-corpus normalisation to mean anything. Normalisation
schemes (min-max over the candidate pool, z-scores) are themselves tunable
knobs, and every one of them is another free parameter to defend.

RRF uses only rank position, so it is scale-free and has a single parameter (k)
with a well-established default of 60. Given that this project's whole argument
is about not hiding free parameters in the evaluation, the fusion method with
one documented constant beats the one with a normalisation policy.

    score(d) = sum over retrievers of 1 / (k + rank(d))
"""

from __future__ import annotations

from ..embedding import Embedder
from ..models import Chunk
from .base import Retriever, ScoredChunk
from .bm25 import BM25Retriever
from .dense import DenseRetriever

RRF_K = 60


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, chunks: list[Chunk], embedder: Embedder, rrf_k: int = RRF_K, pool: int = 50):
        self.chunks = chunks
        self.dense = DenseRetriever(chunks, embedder)
        self.bm25 = BM25Retriever(chunks)
        self.rrf_k = rrf_k
        # Fuse over a deeper pool than the final k: a chunk ranked 30th by one
        # retriever and 2nd by the other should still surface, which cannot
        # happen if each list is truncated at k before fusion.
        self.pool = pool
        self._by_span = {(c.doc_id, c.start, c.end): i for i, c in enumerate(chunks)}

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        fused: dict[int, float] = {}
        for retriever in (self.dense, self.bm25):
            for rank, sc in enumerate(retriever.search(query, k=self.pool), start=1):
                idx = self._by_span[(sc.chunk.doc_id, sc.chunk.start, sc.chunk.end)]
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        order = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [ScoredChunk(self.chunks[i], score) for i, score in order]
