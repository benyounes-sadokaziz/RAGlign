"""Chunker interface and the offset-recovery machinery every chunker shares.

Third-party splitters return strings, not positions. Positions are what this
project is built on, so we recover them ourselves and verify each one against
the source. `recover_offsets` is the seam where a well-behaved splitter is
accepted and a misbehaving one is caught loudly.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from ..models import Chunk, Document, OffsetError


class Chunker(Protocol):
    """A chunking strategy.

    `name` identifies the strategy in results tables and run manifests; it must
    be stable, since results are keyed by it.
    """

    name: str

    def split(self, doc: Document) -> list[Chunk]: ...


def recover_offsets(
    doc: Document,
    pieces: Iterable[str],
    *,
    allow_reordering: bool = False,
) -> list[Chunk]:
    """Locate each piece in `doc.text` and return Chunks with exact offsets.

    Pieces are assumed to appear in document order, so the search advances a
    cursor rather than scanning from zero -- this is both faster and, more
    importantly, disambiguates repeated text (a duplicated code sample matches
    the occurrence that follows the previous chunk, not the first one in the
    file).

    Overlapping chunkers are handled by searching from the *start* of the
    previous chunk rather than its end, so an overlap window can legitimately
    look backwards.

    Raises OffsetError if a piece cannot be located, which means the splitter
    modified text rather than merely cutting it. That is not recoverable by
    fuzzy matching -- it means the strategy needs replacing.
    """
    chunks: list[Chunk] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        idx = doc.text.find(piece, cursor)
        if idx == -1 and allow_reordering:
            idx = doc.text.find(piece)
        if idx == -1:
            raise OffsetError(
                f"piece not found verbatim in {doc.id} at/after {cursor}; "
                f"splitter modified text rather than cutting it.\n"
                f"  piece head: {piece[:100]!r}"
            )
        chunk = Chunk(doc_id=doc.id, start=idx, end=idx + len(piece), text=piece)
        chunk.validate_against(doc)
        chunks.append(chunk)
        # Advance to this chunk's start, not its end: overlapping strategies
        # produce a next chunk that begins before the current one ends.
        cursor = idx
    return chunks


def validate_all(docs: dict[str, Document], chunks: list[Chunk]) -> None:
    """Re-verify every chunk against its document. Cheap insurance, run per config."""
    for chunk in chunks:
        chunk.validate_against(docs[chunk.doc_id])


def coverage_ratio(doc: Document, chunks: list[Chunk]) -> float:
    """Fraction of the document covered by at least one chunk.

    A strategy that silently drops content (e.g. discards code fences or short
    trailing sections) will score well below 1.0 here. Reported alongside
    retrieval metrics because a chunker cannot retrieve what it never indexed --
    and that failure is otherwise invisible in Hit@k.
    """
    if doc.length == 0:
        return 1.0
    covered = bytearray(doc.length)
    for chunk in chunks:
        if chunk.doc_id != doc.id:
            continue
        covered[chunk.start : chunk.end] = b"\x01" * chunk.length
    return sum(covered) / doc.length
