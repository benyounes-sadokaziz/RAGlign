"""Phase 4: rank the grid and explain the recommendation.

Run: .venv/Scripts/python.exe scripts/recommend.py [qa_file] [w_quality w_latency w_context]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from raglign.optimizer import Weights, explain, load_candidates, pareto_frontier, rank


def main(argv: list[str]) -> int:
    qa_file = argv[1] if len(argv) > 1 else "longspan_v1.json"
    weights = Weights(*map(float, argv[2:5])) if len(argv) >= 5 else Weights()

    cands = load_candidates(qa_file)
    if not cands:
        print(f"no runs found for {qa_file}; run scripts/run_grid.py first")
        return 1

    frontier = pareto_frontier(cands)
    ranked = rank(cands, weights)
    front_ids = {c.config_id for c in frontier}

    print(f"qa set : {qa_file}")
    print(f"configs: {len(cands)} evaluated, {len(frontier)} on the Pareto frontier\n")

    header = f"{'':>3} {'config':<26} {'score':>7} {'qual':>6} {'MRR':>6} {'hit@5':>6} {'ms/q':>8} {'chars':>6}  pareto"
    print(header)
    print("-" * len(header))
    for i, (c, score) in enumerate(ranked[:12], 1):
        mark = "  *" if c.config_id in front_ids else ""
        print(
            f"{i:>3} {c.label:<26} {score:>7.3f} {c.quality:>6.3f} {c.mrr:>6.3f} "
            f"{c.hit:>6.3f} {c.latency_ms:>8.1f} {c.chars:>6.0f}{mark}"
        )

    print()
    print("=" * len(header))
    print(explain(ranked[0][0], cands, weights))
    print("=" * len(header))

    print("\nPARETO FRONTIER (nothing here is strictly worse than anything else)")
    for c in sorted(frontier, key=lambda c: -c.quality):
        print(f"  {c.label:<26} quality {c.quality:.3f} | {c.latency_ms:>7.1f} ms | {c.chars:>5.0f} chars")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
