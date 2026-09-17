"""Phase 1: compare chunking strategies on span-aligned retrieval metrics.

Three chunkers, one embedding model, dense retrieval, identical question set.
All scoring flows through the alignment layer, so the comparison is fair across
strategies that carve the corpus differently (TC-1).

Run: .venv/Scripts/python.exe scripts/run_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from raglign.embedding import Embedder
from raglign.experiment import ConfigSpec, evaluate, save_run
from raglign.loader import corpus_fingerprint, index_by_id, load_documents
from raglign.qa import QA_ROOT, load_qa_set

SPECS = [
    ConfigSpec(chunker="fixed", chunker_params={"size": 800, "overlap": 100}),
    ConfigSpec(chunker="recursive", chunker_params={"size": 800, "overlap": 100}),
    ConfigSpec(chunker="heading", chunker_params={"max_size": 1200, "min_size": 200}),
]

K = 5
TAU = 0.5


def main(argv: list[str]) -> int:
    qa_name = argv[1] if len(argv) > 1 else "handwritten_v1.json"
    docs = load_documents()
    qa = load_qa_set(QA_ROOT / qa_name, index_by_id(docs))
    if qa.rejections:
        print(f"refusing to run: {len(qa.rejections)} unresolved QA items")
        return 1

    print(f"corpus    : {len(docs)} docs, fingerprint {corpus_fingerprint(docs)}")
    print(f"qa set    : {qa_name}")
    print(f"questions : {len(qa)} ({sum(len(i.spans) for i in qa.items)} ground-truth spans)")
    print(f"scoring   : coverage overlap, k={K}, tau={TAU}\n")

    embedder = Embedder()
    results = []
    for spec in SPECS:
        print(f"  running {spec.id} ...", end=" ", flush=True)
        result = evaluate(spec, docs, qa.items, embedder=embedder)
        path = save_run(result, docs, qa_name)
        results.append(result)
        print(f"done ({result.build_seconds:.1f}s build) -> runs/{path.name}")

    print()
    header = (
        f"{'config':<28} {'chunks':>7} {'hit@5':>7} {'rec@5':>7} {'MRR':>7} "
        f"{'nDCG':>7} {'soft':>7} {'chars':>7} {'ms':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        m = r.metric(K, TAU)
        print(
            f"{r.spec.id:<28} {r.n_chunks:>7} {m.hit_at_k:>7.3f} {m.recall_at_k:>7.3f} "
            f"{m.mrr:>7.3f} {m.ndcg_at_k:>7.3f} {m.soft_score:>7.3f} "
            f"{m.mean_chars_retrieved:>7.0f} {r.query_ms_mean:>6.1f}"
        )

    print("\nthreshold sensitivity (hit@5 -- TC-8: is the ranking stable in tau?)")
    taus = sorted({t for key in results[0].metrics for t in [float(key.split("tau=")[1])]})
    print(f"{'config':<28}" + "".join(f"{'tau=' + str(t):>9}" for t in taus))
    for r in results:
        row = "".join(f"{r.metric(K, t).hit_at_k:>9.3f}" for t in taus)
        print(f"{r.spec.id:<28}{row}")

    print("\noracle reachability (ceiling from segmentation alone, ignoring the retriever)")
    print(f"{'config':<28}" + "".join(f"{'tau=' + str(t):>9}" for t in taus))
    for r in results:
        row = "".join(f"{r.oracle[t]:>9.3f}" for t in taus)
        print(f"{r.spec.id:<28}{row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
