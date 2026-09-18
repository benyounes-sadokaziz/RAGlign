"""Rank the grid for one corpus, with confidence intervals, and explain the winner.

Run: .venv/Scripts/python.exe scripts/recommend.py --corpus prose [w_q w_lat w_ctx]
"""

from __future__ import annotations

import sys

from _cli import parser, resolve_qa

from raglign import corpora
from raglign.diagnosis import recommended_action, diagnose_config
from raglign.optimizer import Weights, explain, load_candidates, pareto_frontier, rank


def main() -> int:
    p = parser(__doc__)
    p.add_argument("weights", nargs="*", type=float, help="quality latency context")
    args = p.parse_args()

    qa_file = resolve_qa(args.corpus, args.qa)
    weights = Weights(*args.weights[:3]) if len(args.weights) >= 3 else Weights()
    c = corpora.get(args.corpus)

    cands = load_candidates(qa_file, args.corpus)
    if not cands:
        print(f"no runs found for {args.corpus}/{qa_file}; run scripts/run_grid.py first")
        return 1

    frontier = pareto_frontier(cands)
    ranked = rank(cands, weights)
    front_ids = {x.config_id for x in frontier}

    print(f"corpus : {c.name} -- {c.structure}")
    print(f"qa set : {qa_file}")
    print(f"configs: {len(cands)} evaluated, {len(frontier)} on the Pareto frontier\n")

    ks = ranked[0][0].ks or [5]
    hit_cols = "".join(f"{'hit@' + str(k):>8}" for k in ks)
    mrr_cols = "".join(f"{'MRR@' + str(k):>8}" for k in ks)
    header = (
        f"{'':>3} {'config':<26} {'score':>7}{hit_cols}{mrr_cols}"
        f"{'nDCG@5':>8}{'soft':>7}{'ms/q':>9}{'chars':>7}  pareto"
    )
    print(header)
    print("-" * len(header))
    for i, (cd, score) in enumerate(ranked[:12], 1):
        mark = "  *" if cd.config_id in front_ids else ""

        def cell(name: str, k: int) -> str:
            v = cd.metric_at(name, k)
            return f"{v:>8.3f}" if v is not None else f"{'-':>8}"

        print(
            f"{i:>3} {cd.label:<26} {score:>7.3f}"
            + "".join(cell("hit_at_k", k) for k in ks)
            + "".join(cell("mrr", k) for k in ks)
            + f"{cd.ndcg:>8.3f}{cd.soft:>7.3f}{cd.latency_ms:>9.1f}{cd.chars:>7.0f}{mark}"
        )

    print()
    print("=" * len(header))
    print(explain(ranked[0][0], cands, weights))

    winner = ranked[0][0]
    if winner.per_question:
        action = recommended_action(diagnose_config(winner.per_question))
        if action:
            print(f"\nTop fix     {action}")
            print("            (scripts/diagnose.py for the per-question breakdown)")
    print("=" * len(header))

    print("\nPARETO FRONTIER (nothing here is strictly worse than anything else)")
    for cd in sorted(frontier, key=lambda x: -x.quality):
        print(f"  {cd.label:<26} quality {cd.quality:.3f} | {cd.latency_ms:>8.1f} ms | {cd.chars:>5.0f} chars")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
