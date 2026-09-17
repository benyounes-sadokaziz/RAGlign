"""Cross-encoder reranking.

Wraps any retriever: fetch a deeper candidate pool cheaply, then re-score the
pool with a model that reads (query, chunk) jointly instead of comparing two
independently-computed vectors. That joint reading is the whole point -- it can
tell "how do I configure Swagger UI" from "how do I customise the docs assets",
which bi-encoders routinely conflate.

A reranker can only reorder what the first stage retrieved, so it moves MRR and
nDCG much more than hit@k. That distinction is worth keeping visible in the
results: a config whose hit@k is unchanged but whose MRR jumps is doing real
work, and a metric suite reporting only hit@k would call it a no-op.

Cost is the tradeoff and it is measured, not assumed: reranking runs the model
`pool` times per query, so query latency is expected to rise by an order of
magnitude. The optimizer weighs that against the quality gain.
"""

from __future__ import annotations

import os

from ..embedding import MODEL_CACHE, embedding_text
from .base import Retriever, ScoredChunk

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever(Retriever):
    def __init__(self, base: Retriever, model_name: str = DEFAULT_RERANKER, pool: int = 30):
        self.base = base
        self.model_name = model_name
        self.pool = pool
        self.name = f"{getattr(base, 'name', 'base')}+rerank"
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self._model = None

    def _ensure(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self.model_name, cache_dir=str(MODEL_CACHE))
        return self._model

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        candidates = self.base.search(query, k=self.pool)
        if not candidates:
            return []
        model = self._ensure()
        scores = list(model.rerank(query, [embedding_text(c.chunk) for c in candidates]))
        ranked = sorted(zip(candidates, scores), key=lambda p: -p[1])[:k]
        return [ScoredChunk(c.chunk, float(s)) for c, s in ranked]
