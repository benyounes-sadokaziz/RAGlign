"""Fixed-size character windows with overlap.

The naive baseline. Offsets are exact by construction -- there is no splitter to
distrust -- which also makes this the control that proves `recover_offsets` is
not itself introducing error: the two paths must agree on identical inputs.
"""

from __future__ import annotations

from ..models import Chunk, Document


class FixedSizeChunker:
    def __init__(self, size: int = 800, overlap: int = 100):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed_{size}_{overlap}"

    def split(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.size - self.overlap
        for start in range(0, max(doc.length, 1), step):
            end = min(start + self.size, doc.length)
            text = doc.text[start:end]
            if text.strip():
                chunks.append(
                    Chunk(doc_id=doc.id, start=start, end=end, text=text, meta={"strategy": self.name})
                )
            if end >= doc.length:
                break
        return chunks
