"""Core data types.

Deliberately plain dataclasses, not pydantic models: these are constructed in
hot loops (one Chunk per ~500 chars of corpus, per config) and carry no
untrusted input. Pydantic is reserved for on-disk schemas in `schemas.py`,
where validation of hand-authored JSON actually earns its cost.

The invariant that the whole project rests on lives here, in Chunk.__post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


class OffsetError(ValueError):
    """Raised when a chunk's offsets do not address its own text in the source.

    This is a hard failure on purpose. Desynchronised offsets do not crash
    anything downstream -- they silently produce plausible, wrong metrics.
    """


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range [start, end) within one document."""

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span {self.start}:{self.end} in {self.doc_id}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap_chars(self, other: "Span") -> int:
        """Number of characters shared with `other`. Zero across documents."""
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True, slots=True)
class Document:
    """A source document, held exactly as read. Never normalised.

    Any transformation of `text` after load would invalidate every character
    offset derived from it, so there is no method here that returns a modified
    document.
    """

    id: str
    path: str
    text: str

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous slice of one document, produced by a chunking strategy.

    `text` is stored rather than sliced on demand because it is read many times
    (embedding, BM25, reranking, display) and the duplication is a few MB.
    """

    doc_id: str
    start: int
    end: int
    text: str
    # Free-form provenance from the chunker (e.g. heading path). Never used in
    # scoring -- purely for explaining results to a human.
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def span(self) -> Span:
        return Span(self.doc_id, self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start

    def validate_against(self, doc: Document) -> None:
        """Assert the offsets actually address this chunk's text in `doc`.

        Called for every chunk of every strategy. The cost is a string compare;
        the alternative is trusting a third-party splitter to preserve offsets
        it never promised to preserve.
        """
        if doc.id != self.doc_id:
            raise OffsetError(f"chunk doc_id {self.doc_id!r} != document {doc.id!r}")
        actual = doc.text[self.start : self.end]
        if actual != self.text:
            raise OffsetError(
                f"offset desync in {self.doc_id} at {self.start}:{self.end}\n"
                f"  source slice: {actual[:80]!r}\n"
                f"  chunk text  : {self.text[:80]!r}"
            )


@dataclass(frozen=True, slots=True)
class QAItem:
    """One evaluation question with span-level ground truth.

    `spans` are resolved from verbatim quotes at load time (see qa.py); by the
    time a QAItem exists, every span has been confirmed to address the quote
    exactly in the corpus.
    """

    id: str
    question: str
    answer: str
    spans: tuple[Span, ...]
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def doc_ids(self) -> set[str]:
        return {s.doc_id for s in self.spans}


def total_chars(chunks: Iterable[Chunk]) -> int:
    return sum(c.length for c in chunks)
