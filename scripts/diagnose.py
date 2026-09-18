"""Per-question failure root-cause report.

Metrics say a question failed. This says which stage lost it, and the stages
have different fixes -- a chunker that destroyed the evidence cannot be rescued
by any embedding model.

Run: .venv/Scripts/python.exe scripts/diagnose.py --corpus prose
"""

from __future__ import annotations

from collections import Counter

from _cli import parser, resolve_qa

from raglign.diagnosis import Cause, diagnose_config, recommended_action, summarise
from raglign.optimizer import load_candidates, rank


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--top", type=int, default=6, help="how many configs to break down")
    args = p.parse_args()

    qa_file = resolve_qa(args.corpus, args.qa)
    cands = load_candidates(qa_file, args.corpus)
    if not cands:
        print(f"no runs for {args.corpus}/{qa_file}; run scripts/run_grid.py first")
        return 1

    # Non-reranked twin of each config: needed to tell "the reranker demoted it"
    # apart from "the first stage never retrieved it". Without the baseline both
    # look identical from the outcome alone.
    baselines = {
        (c.chunker, c.retriever): c.per_question for c in cands if not c.reranker
    }

    ranked = [c for c, _ in rank(cands)]
    causes = [Cause.CHUNKER_DESTROYED, Cause.CHUNKER_DEGRADED, Cause.RERANKER_DEMOTED,
              Cause.RETRIEVER_MISSED, Cause.RANKED_LOW, Cause.OK]

    print(f"corpus: {args.corpus} | qa: {qa_file} | {len(cands)} configs\n")
    labels = {
        Cause.CHUNKER_DESTROYED: "destroyed",
        Cause.CHUNKER_DEGRADED: "degraded",
        Cause.RERANKER_DEMOTED: "demoted",
        Cause.RETRIEVER_MISSED: "unranked",
        Cause.RANKED_LOW: "low",
        Cause.OK: "ok",
    }
    head = f"{'config':<26}" + "".join(f"{labels[c]:>11}" for c in causes)
    print(head)
    print("-" * len(head))

    fleet: Counter = Counter()
    for c in ranked:
        if not c.per_question:
            continue
        base = baselines.get((c.chunker, c.retriever)) if c.reranker else None
        findings = diagnose_config(c.per_question, base)
        counts = summarise(findings)
        fleet.update(counts)
        print(f"{c.label:<26}" + "".join(f"{counts.get(cz, 0):>11}" for cz in causes))

    print("\nACROSS ALL CONFIGS")
    total = sum(fleet.values()) or 1
    for cz in causes:
        n = fleet.get(cz, 0)
        if n:
            print(f"  {cz.value:<20} {n:>5}  ({n / total:>5.1%})")

    print(f"\nPER-QUESTION BREAKDOWN -- top {args.top} configs\n")
    for c in ranked[: args.top]:
        if not c.per_question:
            continue
        base = baselines.get((c.chunker, c.retriever)) if c.reranker else None
        findings = diagnose_config(c.per_question, base)
        problems = [f for f in findings if f.cause is not Cause.OK]
        print(f"  {c.label}  (MRR {c.mrr:.3f})")
        if not problems:
            print("    every question answered at rank 1")
        for f in problems:
            print(f"    {f.question_id:<6} {f.cause.value:<18} {f.detail}")
        action = recommended_action(findings)
        if action:
            print(f"    -> {action}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
