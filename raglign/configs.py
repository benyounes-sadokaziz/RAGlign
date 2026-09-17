"""The search space.

A full factorial grid over chunking x retrieval x reranking, enumerated
explicitly rather than searched adaptively. At 24 configs an exhaustive sweep
runs in minutes and every cell is observed, so the results table is a complete
map instead of a trajectory through one. Bayesian optimisation over a space this
small would add a surrogate model, an acquisition function and a random seed --
three things to explain and defend -- to save a few minutes of CPU.

Chunk sizes are chosen to bracket the evidence: the long-span question set has
320-533 character spans, so 800/1200-char chunks generally contain a whole span
while 300-char chunks cannot. That bracket is deliberate. It is the regime where
the overlap threshold tau stops being inert and where oracle reachability starts
separating "the retriever ranked it badly" from "the chunker destroyed it".
"""

from __future__ import annotations

from .experiment import ConfigSpec

CHUNKER_VARIANTS = [
    ("fixed", {"size": 800, "overlap": 100}),
    ("fixed", {"size": 300, "overlap": 50}),  # smaller than the evidence, on purpose
    ("recursive", {"size": 800, "overlap": 100}),
    ("heading", {"max_size": 1200, "min_size": 200}),
]

RETRIEVERS = ["dense", "bm25", "hybrid"]
RERANKERS = [None, "Xenova/ms-marco-MiniLM-L-6-v2"]


def full_grid() -> list[ConfigSpec]:
    return [
        ConfigSpec(
            chunker=chunker,
            chunker_params=params,
            retriever=retriever,
            reranker=reranker,
        )
        for chunker, params in CHUNKER_VARIANTS
        for retriever in RETRIEVERS
        for reranker in RERANKERS
    ]


def short_id(spec: ConfigSpec) -> str:
    """Compact label for results tables."""
    size = spec.chunker_params.get("size") or spec.chunker_params.get("max_size")
    rr = "+rr" if spec.reranker else ""
    return f"{spec.chunker[:4]}{size}/{spec.retriever[:3]}{rr}"
