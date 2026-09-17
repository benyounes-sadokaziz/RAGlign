"""Config specification, pipeline construction, and the evaluation loop.

A config is data, not code: a small declarative record naming a chunker, a
retriever, and a reranker with their parameters. The grid in `configs.py` is
then just a list of these, and every run manifest on disk fully describes the
pipeline that produced it -- including the corpus fingerprint, so results
computed against a different corpus can never be silently compared.

Every run evaluates the *same* question set against *every* config, and scoring
goes through the alignment layer, so configs with different chunk boundaries are
compared on identical ground truth (TC-1).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .alignment import DEFAULT_THRESHOLDS, OverlapMode, align, oracle_reachability
from .chunking import FixedSizeChunker, HeadingChunker, RecursiveChunker, SemanticChunker
from .embedding import Embedder
from .loader import corpus_fingerprint
from .metrics import MetricSet, compute_metrics
from .models import Chunk, Document, QAItem

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"

CHUNKERS = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "heading": HeadingChunker,
    "semantic": SemanticChunker,
}


@dataclass(frozen=True)
class ConfigSpec:
    """A complete, reproducible description of one RAG pipeline."""

    chunker: str
    chunker_params: dict = field(default_factory=dict)
    retriever: str = "dense"
    retriever_params: dict = field(default_factory=dict)
    reranker: str | None = None
    embed_model: str = "BAAI/bge-small-en-v1.5"

    @property
    def id(self) -> str:
        parts = [self.chunker, *(f"{v}" for v in self.chunker_params.values()), self.retriever]
        if self.retriever_params:
            parts += [f"{v}" for v in self.retriever_params.values()]
        if self.reranker:
            # Model names are namespaced ("Xenova/ms-marco-..."); the id is used
            # as a filename, so keep only the last segment.
            parts.append(f"rr-{self.reranker.rsplit('/', 1)[-1]}")
        return "_".join(str(p) for p in parts)

    def as_dict(self) -> dict:
        return {"id": self.id, **asdict(self)}


@dataclass
class BuiltPipeline:
    spec: ConfigSpec
    chunks: list[Chunk]
    retriever: object
    build_seconds: float
    index_chars: int


def build_chunks(spec: ConfigSpec, docs: Sequence[Document]) -> list[Chunk]:
    chunker = CHUNKERS[spec.chunker](**spec.chunker_params)
    chunks: list[Chunk] = []
    for doc in docs:
        produced = chunker.split(doc)
        for c in produced:
            c.validate_against(doc)  # the invariant, enforced on every run
        chunks.extend(produced)
    return chunks


def build_pipeline(spec: ConfigSpec, docs: Sequence[Document], embedder: Embedder) -> BuiltPipeline:
    t0 = time.perf_counter()
    chunks = build_chunks(spec, docs)

    if spec.retriever == "dense":
        from .retrieval.dense import DenseRetriever

        retriever = DenseRetriever(chunks, embedder)
    elif spec.retriever == "bm25":
        from .retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever(chunks, **spec.retriever_params)
    elif spec.retriever == "hybrid":
        from .retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(chunks, embedder, **spec.retriever_params)
    else:
        raise ValueError(f"unknown retriever: {spec.retriever}")

    if spec.reranker:
        from .retrieval.rerank import RerankingRetriever

        retriever = RerankingRetriever(retriever, model_name=spec.reranker)

    return BuiltPipeline(
        spec=spec,
        chunks=chunks,
        retriever=retriever,
        build_seconds=time.perf_counter() - t0,
        index_chars=sum(c.length for c in chunks),
    )


@dataclass
class RunResult:
    """Everything one config produced, at every (k, threshold) combination."""

    spec: ConfigSpec
    n_chunks: int
    index_chars: int
    build_seconds: float
    query_ms_mean: float
    query_ms_p90: float
    metrics: dict[str, MetricSet]  # "k=5,tau=0.5" -> MetricSet
    oracle: dict[float, float]  # tau -> fraction of spans reachable at all
    per_question: dict[str, float]  # question id -> soft score, for diagnosis

    def metric(self, k: int, threshold: float) -> MetricSet:
        return self.metrics[f"k={k},tau={threshold}"]


def evaluate(
    spec: ConfigSpec,
    docs: Sequence[Document],
    questions: Sequence[QAItem],
    *,
    embedder: Embedder,
    ks: Sequence[int] = (1, 3, 5, 10),
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    mode: OverlapMode = OverlapMode.COVERAGE,
) -> RunResult:
    pipeline = build_pipeline(spec, docs, embedder)
    max_k = max(ks)

    aligned = []
    latencies: list[float] = []
    for q in questions:
        t0 = time.perf_counter()
        scored = pipeline.retriever.search(q.question, k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        aligned.append(align(q.id, [s.chunk for s in scored], q.spans, mode))

    metrics: dict[str, MetricSet] = {}
    for k in ks:
        for tau in thresholds:
            metrics[f"k={k},tau={tau}"] = compute_metrics(aligned, k=k, threshold=tau)

    all_truths = [s for q in questions for s in q.spans]
    oracle = {
        tau: oracle_reachability(pipeline.chunks, all_truths, tau, mode) for tau in thresholds
    }

    latencies.sort()
    return RunResult(
        spec=spec,
        n_chunks=len(pipeline.chunks),
        index_chars=pipeline.index_chars,
        build_seconds=pipeline.build_seconds,
        query_ms_mean=sum(latencies) / len(latencies),
        query_ms_p90=latencies[int(0.9 * (len(latencies) - 1))],
        metrics=metrics,
        oracle=oracle,
        per_question={a.question_id: a.soft_score(max_k) for a in aligned},
    )


def save_run(result: RunResult, docs: Sequence[Document], qa_file: str, out_dir: Path = RUNS_ROOT) -> Path:
    """Persist one run as JSON (TC-9).

    The corpus fingerprint is recorded because character spans are only valid
    against the exact corpus they were resolved on; a fingerprint mismatch means
    two runs are not comparable, and this is what makes that detectable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{stamp}_{result.spec.id}.json"
    payload = {
        "config": result.spec.as_dict(),
        "corpus_fingerprint": corpus_fingerprint(list(docs)),
        "qa_file": qa_file,
        "n_chunks": result.n_chunks,
        "index_chars": result.index_chars,
        "build_seconds": round(result.build_seconds, 3),
        "query_ms_mean": round(result.query_ms_mean, 3),
        "query_ms_p90": round(result.query_ms_p90, 3),
        "metrics": {key: m.as_dict() for key, m in result.metrics.items()},
        "oracle_reachability": result.oracle,
        "per_question_soft": result.per_question,
        "env": {"python": platform.python_version(), "platform": platform.platform()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
