"""Config ranking and the recommendation explainer.

Three deliberate constraints:

1. **Quality, latency and context cost are separate axes.** Collapsing them into
   one number up front hides the tradeoff that the user actually has to make.
   The Pareto frontier is computed first and reported; the weighted score is
   applied afterwards, only to break ties among non-dominated options.

2. **Context cost is measured in characters retrieved, not in dollars.** This
   project runs local models, so a dollar figure would be invented. Characters
   sent to the generator is the real driver of both token spend and latency
   downstream, it is measured directly, and it does not rot when prices change.

3. **The explanation is generated from the numbers, not by an LLM.** Templated
   text over actual metric deltas cannot hallucinate a reason, and it stays
   correct when the data changes. An LLM asked to explain a results table would
   produce more fluent prose and occasionally invent a cause.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class Candidate:
    """One config's summary, flattened out of a saved run manifest."""

    config_id: str
    label: str
    chunker: str
    retriever: str
    reranker: str | None
    n_chunks: int
    hit: float
    recall: float
    mrr: float
    ndcg: float
    soft: float
    chars: float
    latency_ms: float
    oracle: dict[str, float]

    @property
    def quality(self) -> float:
        """Composite retrieval quality.

        MRR-weighted because on a corpus where most configs eventually find the
        evidence, *where* it lands in the ranking is the difference that reaches
        the generator's context window. hit@k alone saturates and stops
        discriminating (observed directly on the short-span question set).
        """
        return 0.5 * self.mrr + 0.3 * self.ndcg + 0.2 * self.hit


def load_candidates(qa_file: str, runs_root: Path = RUNS_ROOT, k: int = 5, tau: float = 0.5) -> list[Candidate]:
    """Load the most recent run per config for one QA set and corpus.

    Runs from a different corpus fingerprint are excluded rather than merged:
    character spans are only valid against the corpus they were resolved on, so
    mixing fingerprints would compare numbers that do not mean the same thing.
    """
    by_config: dict[str, tuple[str, dict]] = {}
    fingerprints: set[str] = set()

    for path in sorted(runs_root.glob("*.json")):
        if path.name == "validation_study.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("qa_file") != qa_file:
            continue
        fingerprints.add(data["corpus_fingerprint"])
        cid = data["config"]["id"]
        # Filenames are timestamp-prefixed, so the lexicographic max is newest.
        if cid not in by_config or path.name > by_config[cid][0]:
            by_config[cid] = (path.name, data)

    if len(fingerprints) > 1:
        raise ValueError(
            f"runs span {len(fingerprints)} corpus fingerprints: {fingerprints}. "
            "Spans are corpus-specific; re-run the grid rather than comparing across corpora."
        )

    out: list[Candidate] = []
    for cid, (_, data) in by_config.items():
        m = data["metrics"][f"k={k},tau={tau}"]
        cfg = data["config"]
        size = cfg["chunker_params"].get("size") or cfg["chunker_params"].get("max_size")
        rr = "+rerank" if cfg["reranker"] else ""
        out.append(
            Candidate(
                config_id=cid,
                label=f"{cfg['chunker']}{size}/{cfg['retriever']}{rr}",
                chunker=cfg["chunker"],
                retriever=cfg["retriever"],
                reranker=cfg["reranker"],
                n_chunks=data["n_chunks"],
                hit=m["hit_at_k"],
                recall=m["recall_at_k"],
                mrr=m["mrr"],
                ndcg=m["ndcg_at_k"],
                soft=m["soft_score"],
                chars=m["mean_chars_retrieved"],
                latency_ms=data["query_ms_mean"],
                oracle=data["oracle_reachability"],
            )
        )
    return out


def pareto_frontier(cands: Sequence[Candidate]) -> list[Candidate]:
    """Non-dominated configs on (quality up, latency down, chars down).

    A config is dominated when another is at least as good on all three axes and
    strictly better on one. Everything dominated is a strictly worse deal and
    never worth recommending regardless of how the axes are weighted.
    """
    frontier: list[Candidate] = []
    for c in cands:
        dominated = any(
            o is not c
            and o.quality >= c.quality
            and o.latency_ms <= c.latency_ms
            and o.chars <= c.chars
            and (o.quality > c.quality or o.latency_ms < c.latency_ms or o.chars < c.chars)
            for o in cands
        )
        if not dominated:
            frontier.append(c)
    return frontier


def _norm(values: Sequence[float], lower_is_better: bool = False) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    scaled = [(v - lo) / (hi - lo) for v in values]
    return [1.0 - s for s in scaled] if lower_is_better else scaled


@dataclass
class Weights:
    quality: float = 0.7
    latency: float = 0.2
    context: float = 0.1


def rank(cands: Sequence[Candidate], weights: Weights = Weights()) -> list[tuple[Candidate, float]]:
    """Weighted ranking over min-max normalised axes.

    Latency is normalised on a log scale: the observed range spans 0.4 ms to
    ~2900 ms, and on a linear scale every non-reranked config would compress
    into an indistinguishable band near zero, making the axis meaningless for
    every comparison that does not involve a reranker.
    """
    import math

    q = _norm([c.quality for c in cands])
    lat = _norm([math.log10(max(c.latency_ms, 0.01)) for c in cands], lower_is_better=True)
    ctx = _norm([c.chars for c in cands], lower_is_better=True)

    scored = [
        (c, weights.quality * q[i] + weights.latency * lat[i] + weights.context * ctx[i])
        for i, c in enumerate(cands)
    ]
    return sorted(scored, key=lambda p: -p[1])


def explain(winner: Candidate, cands: Sequence[Candidate], weights: Weights = Weights()) -> str:
    """Generate the 'why this config' report from measured deltas only."""
    others = [c for c in cands if c is not winner]
    lines: list[str] = []

    lines.append(f"RECOMMENDED: {winner.label}")
    lines.append("")
    lines.append(
        f"Quality  MRR {winner.mrr:.3f} | hit@5 {winner.hit:.3f} | nDCG {winner.ndcg:.3f}"
    )
    lines.append(
        f"Cost     {winner.latency_ms:.1f} ms/query | {winner.chars:.0f} chars of context | "
        f"{winner.n_chunks} chunks indexed"
    )
    lines.append("")

    # Attribute the win one axis at a time, holding the other two fixed.
    # Comparing against the best config that differs on *any* axis would credit
    # (or blame) the wrong component -- e.g. reporting that dense "trails
    # hybrid" when the config it lost to also added a reranker.
    def best_varying(axis: str) -> Candidate | None:
        pool = [
            c
            for c in others
            if (c.chunker != winner.chunker if axis == "chunker" else c.chunker == winner.chunker)
            and (
                c.retriever != winner.retriever
                if axis == "retriever"
                else c.retriever == winner.retriever
            )
            and bool(c.reranker) == bool(winner.reranker)
        ]
        return max(pool, key=lambda c: c.mrr) if pool else None

    alt_chunker = best_varying("chunker")
    if alt_chunker:
        d = winner.mrr - alt_chunker.mrr
        verb = "outperforms" if d >= 0 else "trails"
        lines.append(
            f"Chunking   {winner.chunker} {verb} the best alternative chunker "
            f"({alt_chunker.label}) by {abs(d):.3f} MRR, holding retrieval fixed."
        )

    alt_retr = best_varying("retriever")
    if alt_retr:
        d = winner.mrr - alt_retr.mrr
        verb = "outperforms" if d >= 0 else "trails"
        lines.append(
            f"Retrieval  {winner.retriever} {verb} the best alternative retriever "
            f"({alt_retr.label}) by {abs(d):.3f} MRR, holding chunking fixed."
        )

    twin = next(
        (
            c
            for c in others
            if c.chunker == winner.chunker
            and c.retriever == winner.retriever
            and bool(c.reranker) != bool(winner.reranker)
        ),
        None,
    )
    if twin:
        dq = winner.mrr - twin.mrr
        dl = winner.latency_ms - twin.latency_ms
        if winner.reranker:
            lines.append(
                f"Reranking  buys {dq:+.3f} MRR for {dl:+.0f} ms/query "
                f"({dq / dl * 1000:+.3f} MRR per second spent)."
            )
        else:
            lines.append(
                f"Reranking  was rejected: it would add {-dl:.0f} ms/query for {-dq:+.3f} MRR."
            )

    worst_oracle = min(float(v) for v in winner.oracle.values())
    if worst_oracle < 0.99:
        lines.append(
            f"WARNING    segmentation ceiling: only {worst_oracle:.1%} of evidence spans are "
            f"reachable at the strictest threshold. Some questions are unanswerable for this "
            f"chunker no matter how good the retriever is."
        )

    lines.append("")
    lines.append(
        f"Weights: quality {weights.quality:.0%}, latency {weights.latency:.0%}, "
        f"context {weights.context:.0%}. Change them to change the recommendation."
    )
    return "\n".join(lines)
