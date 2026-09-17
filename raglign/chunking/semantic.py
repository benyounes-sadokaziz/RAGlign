"""Semantic chunking: cut where the topic changes, not where the ruler says.

The other three strategies decide boundaries from things visible without
understanding the text -- character counts (fixed), whitespace (recursive),
markdown markers (heading). This one embeds each sentence, walks the document,
and cuts where consecutive sentences stop being about the same thing.

    embed each sentence  ->  distance between neighbours  ->  cut at the peaks

A "peak" is a distance above a percentile of that document's own distances,
not a fixed number. Absolute cosine-distance thresholds do not transfer between
documents: a dense API reference has a narrower distance distribution than a
discursive tutorial, and a threshold tuned on one silently over- or under-cuts
the other. A percentile adapts per document and keeps one interpretable knob
("cut at the most topically abrupt 10% of sentence boundaries").

Two deliberate constraints:

1. **Fenced code blocks are atomic.** A code sample is one retrieval unit; its
   lines are not sentences and their embeddings are near-meaningless
   individually. Cutting inside one would also strand the explanation from the
   example it explains -- the most common real retrieval failure on API docs.

2. **Cuts land only on sentence boundaries that exist in the source**, so every
   chunk remains an exact slice and the offset invariant holds unchanged.

Cost note: this embeds every sentence in the corpus, once, on top of embedding
the chunks themselves. It is by far the most expensive strategy to build
(~4x the others here). That cost is real and is reported as build_seconds --
a practitioner choosing this strategy should see what it costs to index.
"""

from __future__ import annotations

import re

import numpy as np

from ..models import Chunk, Document

# Sentence-ish boundary: terminator, closing quote/bracket, then whitespace.
# Deliberately not an NLP sentence splitter -- this text is full of "e.g.",
# "0.95.1" and "app.routes", and a heavyweight tokenizer buys little here while
# adding a dependency. Over-splitting is harmless: adjacent pieces that are
# topically similar get merged back together by the distance step anyway.
_SENT_END = re.compile(r'(?<=[.!?])["\')\]]*\s+')
_FENCE = re.compile(r"^[ \t]*(```|~~~)", re.MULTILINE)


def _atomic_regions(text: str) -> list[tuple[int, int]]:
    """Character ranges that must never be cut through (fenced code blocks)."""
    regions: list[tuple[int, int]] = []
    open_at: int | None = None
    for m in _FENCE.finditer(text):
        if open_at is None:
            open_at = m.start()
        else:
            line_end = text.find("\n", m.end())
            regions.append((open_at, line_end + 1 if line_end != -1 else len(text)))
            open_at = None
    if open_at is not None:  # unterminated fence: protect to end of document
        regions.append((open_at, len(text)))
    return regions


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans covering the document, split at sentence ends.

    Spans are contiguous and cover the text exactly, so any run of consecutive
    spans is itself an exact slice of the document.
    """
    atomic = _atomic_regions(text)

    def inside_atomic(pos: int) -> bool:
        return any(a <= pos < b for a, b in atomic)

    cuts = [0]
    for m in _SENT_END.finditer(text):
        if not inside_atomic(m.start()):
            cuts.append(m.end())
    # A code block is its own unit: cut immediately before and after each one.
    for a, b in atomic:
        cuts.extend([a, b])
    cuts.append(len(text))

    cuts = sorted(set(c for c in cuts if 0 <= c <= len(text)))
    return [(s, e) for s, e in zip(cuts, cuts[1:]) if e > s]


class SemanticChunker:
    """Cut at topic shifts detected from sentence-embedding distances.

    Args:
        percentile: cut at boundaries whose distance exceeds this percentile of
            the document's own distances. Higher = fewer, larger chunks.
        max_size: hard cap; an over-long run is cut at its most distant internal
            boundary, recursively. Without this, a document with a uniform
            distance profile produces one enormous chunk.
        min_size: runs below this are merged forward, since a two-sentence chunk
            rarely carries enough context to answer anything.
    """

    def __init__(
        self,
        percentile: float = 90.0,
        max_size: int = 1200,
        min_size: int = 200,
        buffer_size: int = 0,
        embedder=None,
    ):
        self.percentile = percentile
        self.max_size = max_size
        self.min_size = min_size
        self.buffer_size = buffer_size
        suffix = f"_b{buffer_size}" if buffer_size else ""
        self.name = f"semantic_{int(percentile)}_{max_size}{suffix}"
        self._embedder = embedder

    def _ensure_embedder(self):
        if self._embedder is None:
            from ..embedding import Embedder

            self._embedder = Embedder()
        return self._embedder

    def _cut_points(self, doc: Document, sents: list[tuple[int, int]]) -> list[int]:
        """Indices into `sents` after which a boundary should be cut."""
        if len(sents) < 3:
            return []

        embedder = self._ensure_embedder()

        if self.buffer_size:
            # Embed each position as a window of its neighbours rather than the
            # bare sentence. This is NOT a cost saving -- it is the same number
            # of calls on slightly longer text. It is a signal-quality fix: a
            # single sentence embeds noisily, and consecutive items in a list
            # ("* Never store plaintext passwords.") each look like their own
            # topic, so the unbuffered version cuts enumerations apart. A window
            # spanning several items embeds as "a list of security rules" --
            # one topic -- which is what the list actually is.
            b = self.buffer_size
            texts = [
                doc.text[sents[max(0, i - b)][0] : sents[min(len(sents) - 1, i + b)][1]]
                for i in range(len(sents))
            ]
        else:
            texts = [doc.text[s:e] for s, e in sents]

        vecs = embedder.encode(texts)
        # Vectors are L2-normalised, so cosine distance is 1 - dot product.
        dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
        if not len(dists):
            return []
        threshold = float(np.percentile(dists, self.percentile))
        return [i for i, d in enumerate(dists) if d >= threshold]

    def _enforce_max(self, doc: Document, groups: list[list[int]], sents) -> list[list[int]]:
        """Recursively split groups that exceed max_size at their widest seam."""
        out: list[list[int]] = []
        for g in groups:
            start, end = sents[g[0]][0], sents[g[-1]][1]
            if end - start <= self.max_size or len(g) == 1:
                out.append(g)
                continue
            # Split at the sentence boundary nearest the middle: the semantic
            # signal has already been spent on this group, so a size-driven cut
            # is the honest fallback rather than pretending to find meaning.
            mid = start + (end - start) // 2
            best = min(range(1, len(g)), key=lambda i: abs(sents[g[i]][0] - mid))
            out.extend(self._enforce_max(doc, [g[:best], g[best:]], sents))
        return out

    def split(self, doc: Document) -> list[Chunk]:
        sents = split_sentences(doc.text)
        if not sents:
            return []

        cuts = set(self._cut_points(doc, sents))

        groups: list[list[int]] = []
        current: list[int] = []
        for i in range(len(sents)):
            current.append(i)
            if i in cuts:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        groups = self._enforce_max(doc, groups, sents)

        # Merge runts forward. Groups are contiguous, so a merge is still one
        # valid slice of the document.
        merged: list[list[int]] = []
        for g in groups:
            size = sents[g[-1]][1] - sents[g[0]][0]
            if merged and size < self.min_size:
                merged[-1] = merged[-1] + g
            else:
                merged.append(g)

        chunks: list[Chunk] = []
        for g in merged:
            start, end = sents[g[0]][0], sents[g[-1]][1]
            text = doc.text[start:end]
            if not text.strip():
                continue
            chunk = Chunk(
                doc_id=doc.id,
                start=start,
                end=end,
                text=text,
                meta={"strategy": self.name, "n_sentences": len(g)},
            )
            chunk.validate_against(doc)
            chunks.append(chunk)
        return chunks
