"""Retrieval metrics computed from aligned results.

All deterministic arithmetic over the alignment layer's output -- no LLM judge
anywhere in this path (TC-3). Two runs of the same config on the same corpus
produce bit-identical numbers, which is what makes the chunking comparison
defensible.

Metric choices, and what each one is for:

  hit@k        did any retrieved chunk in the top k satisfy the threshold?
               The blunt, intuitive one. Saturates quickly -- at k=10 on an
               easy corpus most configs score near 1.0 and look identical.
  recall@k     fraction of ground-truth spans covered within top k.
               Differs from hit@k only on multi-span questions, which is
               exactly where it matters.
  mrr          1 / rank of first hit. Rewards ranking the evidence first, not
               merely somewhere in the list -- the difference a reranker makes.
  ndcg@k       graded, position-discounted. Uses raw overlap as the gain, so a
               chunk containing all the evidence outranks one containing half.
  soft_score   threshold-free mean coverage. The control for tau (TC-8): if the
               ranking under soft_score matches the thresholded ranking, the
               conclusion does not depend on where tau was placed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Sequence

from .alignment import AlignedResult


@dataclass(frozen=True)
class MetricSet:
    threshold: float
    k: int
    n_questions: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    soft_score: float
    mean_chars_retrieved: float
    unanswered: int  # questions where nothing in the top k reached the threshold

    def as_dict(self) -> dict:
        return asdict(self)


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg(result: AlignedResult, k: int) -> float:
    """Graded nDCG using overlap as gain.

    The ideal ranking is this same retrieval's gains sorted descending, so nDCG
    measures ordering quality given what was retrieved. An index-wide ideal
    would conflate ordering with recall; recall is already reported separately.
    """
    gains = list(result.scores[:k])
    if not any(gains):
        return 0.0
    ideal = sorted(gains, reverse=True)
    denom = _dcg(ideal)
    return _dcg(gains) / denom if denom else 0.0


def compute_metrics(results: Sequence[AlignedResult], k: int, threshold: float) -> MetricSet:
    if not results:
        raise ValueError("no results to score")

    hits, recalls, rrs, ndcgs, softs = [], [], [], [], []
    unanswered = 0

    for r in results:
        rank = r.first_hit_rank(threshold, k)
        hits.append(1.0 if rank else 0.0)
        if rank is None:
            unanswered += 1
        rrs.append(1.0 / rank if rank else 0.0)
        recalls.append(r.span_recall(threshold, k))
        ndcgs.append(_ndcg(r, k))
        softs.append(r.soft_score(k))

    return MetricSet(
        threshold=threshold,
        k=k,
        n_questions=len(results),
        hit_at_k=mean(hits),
        recall_at_k=mean(recalls),
        mrr=mean(rrs),
        ndcg_at_k=mean(ndcgs),
        soft_score=mean(softs),
        mean_chars_retrieved=mean(r.chars_retrieved(k) for r in results),
        unanswered=unanswered,
    )
