"""Lexical retrieval (BM25).

Included because technical documentation is a genuinely favourable case for
lexical matching: identifiers like `swagger_ui_parameters`, `X-Accel-Buffering`
or `jsonable_encoder` are exactly the tokens a small embedding model is worst at
-- they are rare, compound, and carry the whole meaning of the query. Dense
retrieval smooths them into a neighbourhood; BM25 matches them exactly.

Tokenisation splits on non-alphanumerics but keeps `_`, `-` and `.` inside
tokens, so `X-Forwarded-For` and `app.routes` survive as units. Splitting them
would discard precisely the signal BM25 is here to capture.
"""

from __future__ import annotations

import re

from ..embedding import embedding_text
from ..models import Chunk
from .base import Retriever, ScoredChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        # Indexes the same text the dense retriever embeds, heading prefix
        # included, so the two are compared on identical inputs.
        self.index = BM25Okapi([tokenize(embedding_text(c)) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in order]
