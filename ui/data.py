"""Everything the views need, computed once and cached.

The views stay declarative: they read from here and render. Keeping the
computation in one place means the recommendation card, the Pareto table and the
scatter cannot disagree about which config won -- they all consume the same
ranked list rather than each re-deriving it.

Confidence intervals are attached here too, so no view can display a point
estimate without having its interval available. At the current sample size that
is the difference between a readable table and a misleading one.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

import pandas as pd
import streamlit as st

from raglign import corpora
from raglign.diagnosis import Cause, diagnose_config
from raglign.optimizer import (
    RUNS_ROOT,
    Candidate,
    Weights,
    load_candidates,
    pareto_frontier,
    rank,
)

from .theme import family_of

CAUSES = [
    Cause.CHUNKER_DESTROYED,
    Cause.CHUNKER_DEGRADED,
    Cause.RERANKER_DEMOTED,
    Cause.RETRIEVER_MISSED,
    Cause.RANKED_LOW,
    Cause.OK,
]


@dataclass
class View:
    """One fully-prepared screen's worth of data."""

    corpus: str
    qa_file: str
    tau: float
    weights: Weights
    candidates: list[Candidate]
    ranked: list[tuple[Candidate, float]]
    frontier_ids: set[str]
    winner: Candidate
    winner_score: float
    baselines: dict[tuple[str, str], dict]
    intervals: dict[str, dict[str, tuple[float, float, float]]]

    @property
    def n_questions(self) -> int:
        return len(self.winner.per_question) or 0


def _bootstrap(values: Sequence[float], iterations: int = 1500, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate with a 95% interval, resampling questions.

    Resampling questions rather than retrievals because retrieval here is
    deterministic: the only sampling uncertainty is which questions ended up in
    the evaluation set.
    """
    if not values:
        return 0.0, 0.0, 0.0
    point = mean(values)
    if len(values) < 3:
        return point, point, point
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(iterations))
    return point, draws[int(0.025 * (iterations - 1))], draws[int(0.975 * (iterations - 1))]


def _per_question_series(cand: Candidate) -> tuple[list[float], list[float]]:
    """(reciprocal ranks, hits) from the saved per-question records."""
    rr, hit = [], []
    for stats in cand.per_question.values():
        rank_ = stats.get("rank")
        rr.append(1.0 / rank_ if rank_ else 0.0)
        hit.append(1.0 if rank_ else 0.0)
    return rr, hit


@st.cache_data(show_spinner=False)
def available_corpora() -> list[tuple[str, str, int]]:
    """(name, structure, doc count) for corpora that actually have runs on disk."""
    out = []
    for name in corpora.names():
        c = corpora.get(name)
        n_docs = len(list(c.path.glob(c.pattern))) if c.path.exists() else 0
        out.append((name, c.structure, n_docs))
    return out


@st.cache_data(show_spinner=False)
def build(corpus: str, qa_file: str, tau: float, wq: float, wl: float, wc: float) -> View | None:
    total = max(wq + wl + wc, 1e-9)
    weights = Weights(wq / total, wl / total, wc / total)

    cands = load_candidates(qa_file, corpus, tau=tau)
    if not cands:
        return None

    ranked = rank(cands, weights)
    frontier = {c.config_id for c in pareto_frontier(cands)}
    winner, winner_score = ranked[0]

    intervals: dict[str, dict[str, tuple[float, float, float]]] = {}
    for c in cands:
        rr, hit = _per_question_series(c)
        intervals[c.config_id] = {"mrr": _bootstrap(rr), "hit": _bootstrap(hit)}

    baselines = {(c.chunker, c.retriever): c.per_question for c in cands if not c.reranker}

    return View(
        corpus=corpus,
        qa_file=qa_file,
        tau=tau,
        weights=weights,
        candidates=cands,
        ranked=ranked,
        frontier_ids=frontier,
        winner=winner,
        winner_score=winner_score,
        baselines=baselines,
        intervals=intervals,
    )


def scatter_frame(view: View) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "config": c.label,
            "family": family_of(c.label),
            "quality": c.quality,
            "mrr": c.mrr,
            "latency_ms": max(c.latency_ms, 0.1),
            "chars": c.chars,
            "reranked": bool(c.reranker),
        }
        for c in view.candidates
    ]
    df = pd.DataFrame(rows)
    front = df[df["config"].isin({c.label for c in view.candidates if c.config_id in view.frontier_ids})]
    return df, front.sort_values("latency_ms")


def family_frame(view: View) -> pd.DataFrame:
    counts = Counter(family_of(c.label) for c in view.candidates)
    return pd.DataFrame(
        [{"family": f, "count": n} for f, n in counts.most_common()]
    )


def diagnosis_frame(view: View, limit: int | None = None) -> pd.DataFrame:
    rows = []
    for c, _ in view.ranked[:limit] if limit else view.ranked:
        if not c.per_question:
            continue
        base = view.baselines.get((c.chunker, c.retriever)) if c.reranker else None
        counts = Counter(f.cause for f in diagnose_config(c.per_question, base))
        for order, cause in enumerate(CAUSES):
            rows.append(
                {
                    "config": c.label,
                    "cause": cause.value,
                    "count": counts.get(cause, 0),
                    "order": order,
                }
            )
    return pd.DataFrame(rows)


def cause_totals(view: View) -> Counter:
    total: Counter = Counter()
    for c, _ in view.ranked:
        if not c.per_question:
            continue
        base = view.baselines.get((c.chunker, c.retriever)) if c.reranker else None
        total.update(f.cause for f in diagnose_config(c.per_question, base))
    return total


def field_average(view: View, attr: str) -> float:
    return mean(getattr(c, attr) for c in view.candidates)


def load_json(corpus: str, name: str) -> dict | None:
    path = RUNS_ROOT / corpus / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def spec_pairs(cand: Candidate) -> list[tuple[str, str]]:
    size = "".join(ch for ch in cand.label.split("/")[0] if ch.isdigit())
    retriever = {"dense": "Vector", "bm25": "BM25", "hybrid": "Hybrid (BM25 + Vector)"}.get(
        cand.retriever, cand.retriever
    )
    return [
        ("Chunking", f"{cand.chunker.title()} ({size})"),
        ("Retrieval", retriever),
        ("Reranker", "ON" if cand.reranker else "OFF"),
    ]
