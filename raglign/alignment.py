"""The alignment layer: deciding whether a retrieved chunk hit the ground truth.

This is the piece the project exists for.

THE PROBLEM
-----------
Retrieval metrics need to know which retrieved unit was "correct". The obvious
encoding -- "chunk #57 is the answer" -- is unusable for comparing chunking
strategies, because chunk #57 only exists inside one strategy's segmentation.
Score strategy B against ground truth authored on strategy A's boundaries and
you are measuring how much B's cut points resemble A's, not how well B retrieves.

THE FIX
-------
Ground truth is a character span in the source document. A retrieved chunk is a
character span in the same document. "Did we retrieve the evidence?" becomes a
question about interval overlap -- which every strategy can be asked identically.

CHOOSING THE OVERLAP MEASURE
----------------------------
Three plausible definitions, and they disagree in ways that matter:

  coverage  = |chunk ∩ truth| / |truth|
      "how much of the evidence did this chunk contain?"
  precision = |chunk ∩ truth| / |chunk|
      "how much of this chunk was evidence?"
  iou       = |chunk ∩ truth| / |chunk ∪ truth|
      symmetric; penalises both misses and bloat.

`coverage` is the default, and the choice is not arbitrary. IoU structurally
penalises large chunks: a 1200-char chunk fully containing a 120-char piece of
evidence scores 0.1 IoU, while a 200-char chunk containing the same evidence
scores 0.6 -- so IoU would rank chunking strategies by chunk size rather than by
whether they retrieved the answer. Since the downstream generator receives the
whole chunk, a chunk that contains all the evidence has done its job regardless
of its size. Chunk bloat is a real cost, but it is a *context-budget* cost, so it
is reported separately (`chars_retrieved`) instead of being smuggled into the
hit decision.

All three are implemented; `coverage` is the default, and the study reports the
sensitivity (see TC-8).

THE THRESHOLD
-------------
"Hit if overlap >= tau" contains a free parameter, so no single tau is trusted:
results are swept over several values and the ranking's stability is itself a
reported finding.

Partial credit (`soft_hit`) is also computed: the raw overlap fraction, without
thresholding. It is threshold-free by construction and serves as a check that
the thresholded conclusions are not an artefact of where the cut was placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .models import Chunk, Span

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)


class OverlapMode(str, Enum):
    COVERAGE = "coverage"
    PRECISION = "precision"
    IOU = "iou"


def overlap_score(chunk: Span, truth: Span, mode: OverlapMode = OverlapMode.COVERAGE) -> float:
    """Overlap between a retrieved chunk's span and a ground-truth span, in [0, 1]."""
    inter = chunk.overlap_chars(truth)
    if inter == 0:
        return 0.0
    if mode is OverlapMode.COVERAGE:
        return inter / truth.length if truth.length else 0.0
    if mode is OverlapMode.PRECISION:
        return inter / chunk.length if chunk.length else 0.0
    union = chunk.length + truth.length - inter
    return inter / union if union else 0.0


def best_overlap(chunk: Chunk, truths: Sequence[Span], mode: OverlapMode = OverlapMode.COVERAGE) -> float:
    """Best overlap of one chunk against any of the item's ground-truth spans.

    Max rather than sum: a chunk that covers one of two required evidence spans
    should not be credited as if it covered both. Whether *all* spans were
    collectively covered is a property of the retrieved set, handled in
    `span_recall`, not of any single chunk.
    """
    if not truths:
        return 0.0
    return max(overlap_score(chunk.span, t, mode) for t in truths)


@dataclass(frozen=True, slots=True)
class AlignedResult:
    """Per-question alignment of one ranked retrieval list against ground truth.

    Stores the full [n_spans][n_ranks] overlap matrix rather than pre-reduced
    scores. Every metric is then a reduction over a rank prefix of that matrix,
    so cutting at k and sweeping tau are both free -- no re-retrieval, and no
    risk of a metric accidentally reading past its own cutoff.
    """

    question_id: str
    # matrix[i][j] = overlap of retrieved chunk j with ground-truth span i
    matrix: tuple[tuple[float, ...], ...]
    chunk_lengths: tuple[int, ...]

    @property
    def n_spans(self) -> int:
        return len(self.matrix)

    @property
    def scores(self) -> tuple[float, ...]:
        """Per retrieved chunk, its best overlap against any ground-truth span."""
        if not self.matrix:
            return ()
        return tuple(max(col) for col in zip(*self.matrix))

    def chars_retrieved(self, k: int | None = None) -> int:
        return sum(self.chunk_lengths[:k] if k else self.chunk_lengths)

    def first_hit_rank(self, threshold: float, k: int | None = None) -> int | None:
        """1-based rank of the first chunk reaching `threshold`, within top k."""
        for i, s in enumerate(self.scores[:k] if k else self.scores):
            if s >= threshold:
                return i + 1
        return None

    def span_best(self, k: int | None = None) -> tuple[float, ...]:
        """Best overlap achieved for each ground-truth span within the top k."""
        return tuple(max(row[:k] if k else row, default=0.0) for row in self.matrix)

    def span_recall(self, threshold: float, k: int | None = None) -> float:
        """Fraction of ground-truth spans covered by some chunk in the top k.

        This is what scores multi-span (multi-hop) questions honestly: retrieving
        the same evidence twice does not count as finding both pieces.
        """
        best = self.span_best(k)
        if not best:
            return 0.0
        return sum(1 for b in best if b >= threshold) / len(best)

    def soft_score(self, k: int | None = None) -> float:
        """Threshold-free partial credit: mean best coverage over ground-truth spans."""
        best = self.span_best(k)
        return sum(best) / len(best) if best else 0.0


def align(
    question_id: str,
    retrieved: Sequence[Chunk],
    truths: Sequence[Span],
    mode: OverlapMode = OverlapMode.COVERAGE,
) -> AlignedResult:
    """Align one ranked list of retrieved chunks against an item's ground truth."""
    matrix = tuple(
        tuple(overlap_score(c.span, t, mode) for c in retrieved) for t in truths
    )
    return AlignedResult(
        question_id=question_id,
        matrix=matrix,
        chunk_lengths=tuple(c.length for c in retrieved),
    )


def oracle_reachability(
    chunks: Iterable[Chunk],
    truths: Sequence[Span],
    threshold: float,
    mode: OverlapMode = OverlapMode.COVERAGE,
) -> float:
    """Fraction of ground-truth spans that *any* chunk in the index could satisfy.

    The ceiling imposed by segmentation alone, independent of the retriever.
    If a strategy cuts an evidence span across two chunks such that neither
    reaches the threshold, that question is unanswerable for that strategy no
    matter how good the embeddings are.

    Reporting this separates two failure modes that Hit@k conflates:
    "the retriever ranked it poorly" versus "the chunker destroyed the evidence".
    It is also the number that most directly exposes why chunk-ID ground truth
    is unfair -- and it costs one pass over the index to compute.
    """
    if not truths:
        return 0.0
    reachable = 0
    for t in truths:
        for c in chunks:
            if c.doc_id != t.doc_id:
                continue
            if overlap_score(c.span, t, mode) >= threshold:
                reachable += 1
                break
    return reachable / len(truths)
