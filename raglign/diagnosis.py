"""Automatic failure root-cause diagnosis.

A retrieval metric says *that* a question failed. It cannot say *which stage*
lost it -- and the stages have completely different fixes:

    chunker destroyed the evidence   -> change chunk size/strategy.
                                        A better embedding model cannot help;
                                        the text the answer needs no longer
                                        exists as a contiguous retrievable unit.
    retriever ranked it too low      -> change embeddings, try hybrid, add a
                                        reranker. The evidence was sitting in
                                        the index the whole time.
    reranker demoted it              -> drop or retune the reranker. The first
                                        stage had it; the second stage pushed
                                        it out.

Standard evaluation reports all three as one number going down, which sends you
tuning the wrong component. Separating them needs per-question *reachability* --
"could any chunk in this index have answered this?" -- and that question is only
askable when ground truth is a span. With chunk-ID ground truth there is no way
to ask whether evidence survived a segmentation that never produced that chunk.

This is the most direct practical payoff of the whole span-alignment approach:
it does not just make the comparison fair, it tells you what to fix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class Cause(str, Enum):
    OK = "ok"
    CHUNKER_DESTROYED = "chunker_destroyed"
    CHUNKER_DEGRADED = "chunker_degraded"
    RETRIEVER_MISSED = "retriever_missed"
    RERANKER_DEMOTED = "reranker_demoted"
    RANKED_LOW = "ranked_low"


FIXES: dict[Cause, str] = {
    Cause.CHUNKER_DESTROYED: "change chunking (size/strategy) -- embeddings cannot recover this",
    Cause.CHUNKER_DEGRADED: "evidence survived only partially: increase chunk size before tuning retrieval",
    Cause.RETRIEVER_MISSED: "evidence was indexed but unranked: try hybrid, or a reranker",
    Cause.RERANKER_DEMOTED: "first stage found it, reranker pushed it out: drop or retune reranking",
    Cause.RANKED_LOW: "found but not first: a reranker is the targeted fix",
    Cause.OK: "",
}


@dataclass(frozen=True)
class Finding:
    question_id: str
    cause: Cause
    detail: str

    @property
    def fix(self) -> str:
        return FIXES[self.cause]


def diagnose_question(
    stats: dict,
    baseline_stats: dict | None = None,
    *,
    reachable_threshold: float = 0.999,
    degraded_threshold: float = 0.999,
    good_rank: int = 1,
) -> Finding:
    """Classify one question's outcome for one config.

    Args:
        stats: this config's per-question record (rank, reachable, best_coverage).
        baseline_stats: the same question under the identical config *without*
            the reranker, when one exists. Required to distinguish "the reranker
            broke it" from "the first stage never had it" -- without the
            baseline those are indistinguishable from the outcome alone.
    """
    qid = str(stats.get("qid", ""))
    rank = stats.get("rank")
    reachable = float(stats.get("reachable", 0.0))
    coverage = float(stats.get("best_coverage", 0.0))
    achievable = float(stats.get("max_achievable", 1.0))

    # Checked first, and unconditionally: if the evidence is not reachable then
    # a good rank would be luck (a chunk that partially overlaps, scoring under
    # tau), and blaming the retriever would send the user to fix the wrong thing.
    if reachable < reachable_threshold:
        return Finding(
            qid,
            Cause.CHUNKER_DESTROYED,
            f"no chunk covers the evidence (best reachable {reachable:.2f}); "
            f"retrieved best coverage {coverage:.2f}",
        )

    # Evidence that survived only in fragments. It clears tau, so `reachable`
    # calls it fine, but no chunk holds the whole answer -- and a partial chunk
    # is both harder to rank and less useful to the generator. Checked before
    # the retriever so that "my chunks are too small" is not reported as "my
    # embeddings are bad", which was the misattribution this cause was added to
    # fix after fixed300 showed zero chunker failures despite collapsing.
    if rank is None and achievable < degraded_threshold:
        return Finding(
            qid,
            Cause.CHUNKER_DEGRADED,
            f"no chunk holds more than {achievable:.0%} of the evidence",
        )

    if rank is None:
        if baseline_stats is not None and baseline_stats.get("rank") is not None:
            return Finding(
                qid,
                Cause.RERANKER_DEMOTED,
                f"first stage had it at rank {baseline_stats['rank']}; after reranking it is gone",
            )
        return Finding(
            qid,
            Cause.RETRIEVER_MISSED,
            f"evidence was indexed and reachable ({reachable:.2f}) but never retrieved",
        )

    if rank > good_rank:
        return Finding(qid, Cause.RANKED_LOW, f"evidence found at rank {rank}, not 1")

    return Finding(qid, Cause.OK, "evidence retrieved at rank 1")


def diagnose_config(
    per_question: dict[str, dict],
    baseline_per_question: dict[str, dict] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    for qid, stats in sorted(per_question.items()):
        base = (baseline_per_question or {}).get(qid)
        findings.append(diagnose_question({**stats, "qid": qid}, base))
    return findings


def summarise(findings: list[Finding]) -> Counter:
    return Counter(f.cause for f in findings)


def recommended_action(findings: list[Finding]) -> str:
    """The single highest-leverage fix for this config, or empty if none.

    Ranked by how much the fix constrains everything downstream: a chunker that
    destroys evidence caps every retriever that could ever run on it, so it is
    reported first even when another cause is more frequent.
    """
    counts = summarise(findings)
    total = sum(counts.values()) or 1
    for cause in (
        Cause.CHUNKER_DESTROYED,
        Cause.CHUNKER_DEGRADED,
        Cause.RERANKER_DEMOTED,
        Cause.RETRIEVER_MISSED,
        Cause.RANKED_LOW,
    ):
        n = counts.get(cause, 0)
        if n:
            return f"{n}/{total} questions: {FIXES[cause]}"
    return ""
