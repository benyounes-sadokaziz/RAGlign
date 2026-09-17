"""Markdown heading-aware chunking.

Written here rather than taken from a library because the available markdown
splitters rebuild text from parsed structure (re-emitting headings, normalising
blank lines), which destroys the verbatim-slice property everything downstream
depends on. This one never builds a string: it only computes cut positions and
slices the document.

The hypothesis this strategy tests: on structured API documentation, a section
under one heading is a semantically coherent retrieval unit, so respecting
heading boundaries should beat cutting every N characters. The corpus was chosen
so this hypothesis can actually fail (see TECHNICAL_CHOICES TC-7).

Two details that matter:

1. **Fenced code is skipped when detecting headings.** A `# comment` line inside
   a Python block is not a heading. Without fence tracking this corpus -- which
   is roughly half code samples -- would shatter into fragments cut mid-example.

2. **Heading context is carried in `meta`, never prepended to `text`.** The
   retrieval benefit of "Section: Query Parameters > Optional parameters" as
   embedding context is real, but splicing it into `text` would break
   `doc.text[start:end] == text`. The embedder reads the prefix from meta
   instead, so the invariant survives and the benefit is kept.
"""

from __future__ import annotations

import re

from ..models import Chunk, Document
from .base import recover_offsets

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")


def _heading_positions(text: str) -> list[tuple[int, int, str]]:
    """Return (offset, level, title) for each ATX heading outside fenced code."""
    out: list[tuple[int, int, str]] = []
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
        elif not in_fence:
            m = _HEADING_RE.match(stripped)
            if m:
                out.append((offset, len(m.group(1)), m.group(2).strip()))
        offset += len(line)
    return out


def _heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


class HeadingChunker:
    """Split at heading boundaries; sub-split oversized sections, merge runts.

    Args:
        max_size: sections longer than this are sub-split by character recursion.
        min_size: sections shorter than this are merged into the previous chunk
            (adjacent sections are contiguous, so a merge is still one valid span).
        sub_overlap: overlap used only when sub-splitting an oversized section.
    """

    def __init__(self, max_size: int = 1200, min_size: int = 200, sub_overlap: int = 100):
        self.max_size = max_size
        self.min_size = min_size
        self.sub_overlap = sub_overlap
        self.name = f"heading_{max_size}_{min_size}"
        self._sub = None  # lazily built; avoids import cost when unused

    def _sub_split(self, doc: Document, start: int, end: int, prefix: str) -> list[Chunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if self._sub is None:
            self._sub = RecursiveCharacterTextSplitter(
                chunk_size=self.max_size,
                chunk_overlap=self.sub_overlap,
                keep_separator=True,
                strip_whitespace=False,
                separators=["\n\n", "\n", " ", ""],
            )
        section = doc.text[start:end]
        pieces = self._sub.split_text(section)
        # Recover offsets within the section, then shift into document space.
        local = recover_offsets(Document(doc.id, doc.path, section), pieces)
        return [
            Chunk(
                doc_id=doc.id,
                start=start + c.start,
                end=start + c.end,
                text=c.text,
                meta={"strategy": self.name, "context_prefix": prefix, "sub_split": True},
            )
            for c in local
        ]

    def split(self, doc: Document) -> list[Chunk]:
        headings = _heading_positions(doc.text)

        # Section boundaries: document start, every heading, document end.
        bounds = [0] + [off for off, _, _ in headings] + [doc.length]
        # Heading path in force for each section (the prelude before the first
        # heading has none).
        paths: list[str] = [""]
        stack: list[tuple[int, str]] = []
        for _, level, title in headings:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            paths.append(_heading_path(stack))

        raw: list[Chunk] = []
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            if start == end or not doc.text[start:end].strip():
                continue
            prefix = paths[i]
            if end - start > self.max_size:
                raw.extend(self._sub_split(doc, start, end, prefix))
            else:
                raw.append(
                    Chunk(
                        doc_id=doc.id,
                        start=start,
                        end=end,
                        text=doc.text[start:end],
                        meta={"strategy": self.name, "context_prefix": prefix},
                    )
                )

        merged = self._merge_runts(doc, raw)
        for c in merged:
            c.validate_against(doc)
        return merged

    def _merge_runts(self, doc: Document, chunks: list[Chunk]) -> list[Chunk]:
        """Absorb sub-minimum chunks into the previous one when contiguous.

        A lone heading line ("## Recap") carries almost no retrievable signal on
        its own but does when attached to the section it follows. Merging is
        skipped across non-contiguous boundaries so spans stay exact slices.
        """
        out: list[Chunk] = []
        for c in chunks:
            if out and c.length < self.min_size and out[-1].end == c.start:
                prev = out[-1]
                out[-1] = Chunk(
                    doc_id=prev.doc_id,
                    start=prev.start,
                    end=c.end,
                    text=doc.text[prev.start : c.end],
                    meta=prev.meta,
                )
            else:
                out.append(c)
        return out
