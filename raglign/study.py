"""The before/after validation study: chunk-ID ground truth vs span alignment.

This is the artifact the project is built to produce. It answers, with numbers:
*does chunking-independent ground truth actually change which config you would
pick?* If the answer were "no", RAGlign's premise would be wrong, and the study
is constructed so that it could come out that way.

HOW CHUNK-ID GROUND TRUTH IS SIMULATED
--------------------------------------
A practitioner builds an eval set by chunking the corpus once -- with whatever
strategy they happen to be using, here called the *author strategy* -- and
recording, per question, which chunk contains the answer. Ground truth is then
a chunk identity.

Later they want to compare a different chunking strategy. The chunk IDs do not
transfer, because the other strategy's chunks are different objects. What they
do in practice is ask whether the retrieved chunk *is* the gold chunk -- matched
by content, since IDs are meaningless across segmentations. That is formalised
here as: a retrieved chunk is a hit if it overlaps the gold *chunk* (not the
gold span) by at least `match_iou`.

IoU against the gold chunk is the fair formalisation. Requiring exact boundary
equality would guarantee every other strategy scores zero -- a strawman. IoU at
0.9 means "essentially the same passage", which is the most generous reading of
chunk-ID matching that still respects what a chunk ID means.

THE BIAS THIS EXPOSES
---------------------
Under chunk-ID ground truth, the author strategy is evaluated against its own
boundaries and matches them perfectly by construction. Every other strategy is
penalised in proportion to how much its cut points differ -- which is a measure
of boundary agreement, not of retrieval quality. Span alignment removes that
term: both strategies are asked the same question, "did you retrieve the text
that answers this?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .alignment import OverlapMode, overlap_score
from .models import Chunk, QAItem, Span


@dataclass(frozen=True)
class GoldChunk:
    """The chunk an annotator would have recorded, working in the author strategy."""

    question_id: str
    span: Span
    coverage: float  # how much of the true evidence that chunk actually held


def build_chunk_id_ground_truth(
    questions: Sequence[QAItem],
    author_chunks: Sequence[Chunk],
) -> dict[str, list[GoldChunk]]:
    """Simulate annotating each question against one strategy's chunks.

    For each ground-truth span, the annotator picks the chunk that best contains
    it. This is a faithful model of the manual process and, notably, it is the
    *most favourable* possible version of it -- a real annotator scanning a list
    of chunks would sometimes pick a worse one.
    """
    gold: dict[str, list[GoldChunk]] = {}
    for q in questions:
        picks: list[GoldChunk] = []
        for truth in q.spans:
            best, best_cov = None, 0.0
            for c in author_chunks:
                if c.doc_id != truth.doc_id:
                    continue
                cov = overlap_score(c.span, truth, OverlapMode.COVERAGE)
                if cov > best_cov:
                    best, best_cov = c, cov
            if best is not None:
                picks.append(GoldChunk(q.id, best.span, best_cov))
        gold[q.id] = picks
    return gold


def chunk_id_hits(
    retrieved: Sequence[Chunk],
    gold: Sequence[GoldChunk],
    match_iou: float = 0.9,
) -> list[bool]:
    """Would chunk-ID matching have counted each retrieved chunk as correct?"""
    out: list[bool] = []
    for c in retrieved:
        hit = any(
            overlap_score(c.span, g.span, OverlapMode.IOU) >= match_iou for g in gold
        )
        out.append(hit)
    return out


@dataclass
class StudyRow:
    config_id: str
    is_author: bool
    naive_hit_at_k: float  # scored against chunk-ID ground truth
    naive_mrr: float
    aligned_hit_at_k: float  # scored against span ground truth
    aligned_mrr: float

    @property
    def hit_delta(self) -> float:
        return self.aligned_hit_at_k - self.naive_hit_at_k
