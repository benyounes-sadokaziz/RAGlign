"""Retriever interface.

Narrow on purpose: `search(query, k) -> ranked ScoredChunks`. Everything the
evaluation layer needs is the ranked list; how it was produced (dense, lexical,
fused, reranked) is invisible to it. That is what lets the experiment grid treat
retrievers as interchangeable and makes swapping in a real vector store a
single-file change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import Chunk


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]: ...
