"""Recursive character splitting (langchain-text-splitters), offsets recovered.

The standard "sensible default" in most RAG stacks, and therefore the strategy
worth beating. It tries paragraph, then line, then word, then character
boundaries, so chunks tend to end at natural breaks -- but it makes no attempt
to respect markdown structure, and it will happily cut a fenced code block in
half.

`keep_separator=True` is not optional here: with separators stripped, the
returned pieces no longer appear verbatim in the source and offset recovery
fails by design rather than silently mis-aligning.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..models import Chunk, Document
from .base import recover_offsets


class RecursiveChunker:
    def __init__(self, size: int = 800, overlap: int = 100):
        self.size = size
        self.overlap = overlap
        self.name = f"recursive_{size}_{overlap}"
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            keep_separator=True,
            strip_whitespace=False,  # stripping would break verbatim recovery
            separators=["\n\n", "\n", " ", ""],
        )

    def split(self, doc: Document) -> list[Chunk]:
        pieces = self._splitter.split_text(doc.text)
        chunks = recover_offsets(doc, pieces)
        return [
            Chunk(c.doc_id, c.start, c.end, c.text, {"strategy": self.name})
            for c in chunks
            if c.text.strip()
        ]
