"""Phase 3: evaluate the full config grid and persist every run.

Run: .venv/Scripts/python.exe scripts/run_grid.py [qa_file]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from raglign.configs import full_grid, short_id
from raglign.embedding import Embedder
from raglign.experiment import evaluate, save_run
from raglign.loader import corpus_fingerprint, index_by_id, load_documents
from raglign.qa import QA_ROOT, load_qa_set

K = 5
TAU = 0.5


def main(argv: list[str]) -> int:
    qa_name = argv[1] if len(argv) > 1 else "longspan_v1.json"
    docs = load_documents()
    qa = load_qa_set(QA_ROOT / qa_name, index_by_id(docs))
    if qa.rejections:
        print(f"refusing to run: {len(qa.rejections)} unresolved QA items")
        return 1

    grid = full_grid()
    print(f"corpus    : {len(docs)} docs, fingerprint {corpus_fingerprint(docs)}")
    print(f"qa set    : {qa_name} ({len(qa)} questions)")
    print(f"grid      : {len(grid)} configs\n")

    embedder = Embedder()
    results = []
    t_start = time.perf_counter()
    for i, spec in enumerate(grid, 1):
        print(f"  [{i:>2}/{len(grid)}] {spec.id:<44}", end=" ", flush=True)
        r = evaluate(spec, docs, qa.items, embedder=embedder)
        save_run(r, docs, qa_name)
        results.append(r)
        m = r.metric(K, TAU)
        print(f"hit@5={m.hit_at_k:.3f} mrr={m.mrr:.3f} {r.query_ms_mean:>6.1f}ms")

    print(f"\ntotal {time.perf_counter() - t_start:.0f}s\n")

    results.sort(key=lambda r: (-r.metric(K, TAU).mrr, -r.metric(K, TAU).hit_at_k))
    header = (
        f"{'config':<22} {'chunks':>7} {'hit@5':>7} {'rec@5':>7} {'MRR':>7} {'nDCG':>7} "
        f"{'soft':>7} {'chars':>6} {'ms/q':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        m = r.metric(K, TAU)
        print(
            f"{short_id(r.spec):<22} {r.n_chunks:>7} {m.hit_at_k:>7.3f} {m.recall_at_k:>7.3f} "
            f"{m.mrr:>7.3f} {m.ndcg_at_k:>7.3f} {m.soft_score:>7.3f} "
            f"{m.mean_chars_retrieved:>6.0f} {r.query_ms_mean:>7.1f}"
        )

    print("\noracle reachability by chunker (segmentation ceiling, retriever-independent)")
    taus = sorted(results[0].oracle)
    seen = set()
    print(f"{'chunker':<22}" + "".join(f"{'tau=' + str(t):>9}" for t in taus))
    for r in results:
        key = (r.spec.chunker, tuple(r.spec.chunker_params.items()))
        if key in seen:
            continue
        seen.add(key)
        label = short_id(r.spec).split("/")[0]
        print(f"{label:<22}" + "".join(f"{r.oracle[t]:>9.3f}" for t in taus))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
