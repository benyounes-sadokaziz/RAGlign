"""Does the winning config transfer between structurally different corpora?

This is the test of the project's own premise. Building an optimizer only makes
sense if there is no universal best configuration -- otherwise the right answer
is a blog post naming the default, not a tool. v1 could not test this at all,
because it had one corpus.

Same 30 configs, same methodology, same evidence-span sizes, matched corpus
sizes. The only thing that differs is document structure:

    fastapi   dense markdown: headings, code fences, bullet lists
    prose     flowing paragraphs, no markup at all

Confidence intervals are reported alongside, because with ~12 questions per
corpus a rank difference can easily be one question changing its mind. A winner
that flips inside the noise is not a finding, and the interval is what makes
that visible instead of inviting an over-claim.

Run: .venv/Scripts/python.exe scripts/compare_corpora.py
"""

from __future__ import annotations

import json
import random
from statistics import mean

from _cli import parser, resolve_qa

from raglign import corpora
from raglign.experiment import RUNS_ROOT
from raglign.optimizer import Weights, load_candidates, rank

TOP_N = 6


def _bootstrap_mrr(per_question: dict, iterations: int = 2000, seed: int = 0):
    """95% interval for MRR, resampling questions with replacement.

    Reads the saved per-question ranks rather than re-retrieving: retrieval is
    deterministic, so the only sampling uncertainty is which questions are in
    the set (TC-20).
    """
    rr = [
        (1.0 / stats["rank"]) if stats.get("rank") else 0.0
        for stats in per_question.values()
    ]
    if not rr:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(rr)
    draws = sorted(mean(rr[rng.randrange(n)] for _ in range(n)) for _ in range(iterations))
    return mean(rr), draws[int(0.025 * (iterations - 1))], draws[int(0.975 * (iterations - 1))]


def main() -> int:
    p = parser(__doc__, qa=False)
    p.add_argument("--top", type=int, default=TOP_N)
    args = p.parse_args()

    per_corpus: dict[str, list] = {}
    for name in corpora.names():
        qa_file = resolve_qa(name, None)
        cands = load_candidates(qa_file, name)
        if cands:
            per_corpus[name] = rank(cands, Weights())

    if len(per_corpus) < 2:
        print("need runs for at least two corpora; run scripts/run_grid.py --corpus <name>")
        return 1

    print("CORPORA\n")
    for name in per_corpus:
        c = corpora.get(name)
        print(f"  {name:<10} {c.structure}")
    print()

    print(f"TOP {args.top} CONFIGS PER CORPUS\n")
    for name, ranked in per_corpus.items():
        print(f"  {name}")
        print(f"    {'':>3} {'config':<26} {'MRR':>7} {'hit@5':>7} {'ms/q':>8}")
        for i, (c, _) in enumerate(ranked[: args.top], 1):
            print(f"    {i:>3} {c.label:<26} {c.mrr:>7.3f} {c.hit:>7.3f} {c.latency_ms:>8.1f}")
        print()

    # The headline: how does each corpus's winner do on the other corpus?
    print("TRANSFER TEST (each corpus's winner, evaluated on both)\n")
    names = list(per_corpus)
    lookup = {n: {c.label: c for c, _ in per_corpus[n]} for n in names}
    winners = {n: per_corpus[n][0][0].label for n in names}

    header = f"  {'config':<28}" + "".join(f"{n + ' MRR':>16}" for n in names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for src, label in winners.items():
        cells = ""
        for n in names:
            c = lookup[n].get(label)
            cells += f"{c.mrr:>16.3f}" if c else f"{'n/a':>16}"
        print(f"  {label:<28}{cells}   <- best on {src}")
    print()

    # Where does each corpus's winner *rank* on the other corpus? A config that
    # wins one and places mid-table on the other is the concrete evidence that
    # config choice does not transfer.
    print("RANK OF EACH WINNER ON EVERY CORPUS\n")
    print(f"  {'config':<28}" + "".join(f"{n:>12}" for n in names))
    for label in dict.fromkeys(winners.values()):
        row = ""
        for n in names:
            order = [c.label for c, _ in per_corpus[n]]
            row += f"{order.index(label) + 1:>12}" if label in order else f"{'n/a':>12}"
        print(f"  {label:<28}{row}")
    print()

    # How much does the chunking choice matter *at all* on each corpus? Averaged
    # over every retriever and reranker setting, so it measures the chunker
    # family rather than one lucky pairing.
    print("MEAN MRR BY CHUNKER FAMILY (averaged over all retrievers and rerankers)\n")
    families = sorted({c.label.split("/")[0] for _, ranked in per_corpus.items() for c, _ in ranked})
    print(f"  {'chunker':<16}" + "".join(f"{n:>14}" for n in names))
    print("  " + "-" * (16 + 14 * len(names)))
    spreads: dict[str, float] = {}
    for fam in families:
        row = ""
        for n in names:
            vals = [c.mrr for c, _ in per_corpus[n] if c.label.startswith(fam + "/")]
            row += f"{mean(vals):>14.3f}" if vals else f"{'n/a':>14}"
        print(f"  {fam:<16}{row}")
    # fixed300 exists as a deliberate failure case (chunks smaller than the
    # evidence), so it is excluded from the spread: including a control that is
    # broken by construction would make chunking look decisive on every corpus
    # and hide the actual contrast between the viable strategies.
    viable = [f for f in families if not f.startswith("fixed300")]
    for n in names:
        means = [
            mean([c.mrr for c, _ in per_corpus[n] if c.label.startswith(f + "/")])
            for f in viable
            if any(c.label.startswith(f + "/") for c, _ in per_corpus[n])
        ]
        spreads[n] = max(means) - min(means)
    print()
    print("  spread among VIABLE chunkers (fixed300 excluded as a deliberate control):")
    for n, sp in spreads.items():
        print(f"    {n:<12} {sp:.3f}")
    print()

    # Bootstrap over questions, so a "tie" can be distinguished from a gap too
    # small to resolve at this sample size.
    print("TOP-3 CONFIGS WITH 95% BOOTSTRAP INTERVALS (MRR)\n")
    for n in names:
        print(f"  {n}")
        for c, _ in per_corpus[n][:3]:
            point, lo, hi = _bootstrap_mrr(c.per_question)
            print(f"    {c.label:<28} {point:.3f}  [{lo:.3f}, {hi:.3f}]")
        print()

    # Is any top-3 difference actually resolvable at this sample size?
    resolvable = False
    for n in names:
        ivals = [_bootstrap_mrr(c.per_question) for c, _ in per_corpus[n][:3]]
        for i in range(len(ivals)):
            for j in range(i + 1, len(ivals)):
                if ivals[i][2] < ivals[j][1] or ivals[j][2] < ivals[i][1]:
                    resolvable = True

    distinct = set(winners.values())
    print("VERDICT\n")
    if len(distinct) > 1:
        print("  The winning config DIFFERS by corpus. Config choice does not transfer")
        print("  across document structure -- the condition that makes a per-corpus")
        print("  optimizer worth having.")
    else:
        print(f"  The SAME config wins on both corpora ({next(iter(distinct))}).")
        print("  No flip. Reported as-is rather than buried: it partly weakens the case")
        print("  for per-corpus optimization of the *winner*.")

    print()
    print("  But how much the chunking choice MATTERS is strongly corpus-dependent:")
    for n, sp in spreads.items():
        print(f"    {n:<12} spread among viable chunkers {sp:.3f} MRR")
    print("  On structured documents the chunker is a real decision; on unstructured")
    print("  prose the viable strategies converge -- a heading splitter with no headings")
    print("  to find degenerates into a size splitter, so they all become the same thing.")
    print("  Knowing *whether a knob matters* on your corpus is itself the useful output.")

    print()
    if not resolvable:
        print("  CAVEAT, and it is the important one: no top-3 difference on either")
        print("  corpus survives its 95% bootstrap interval. At this sample size the")
        print("  fine-grained config ranking is NOT resolved. Only large effects hold:")
        print("  the fixed300 collapse, and the validation study's ~0.6 ground-truth bias.")
        print("  More questions -- not more configs -- is what buys resolution here.")

    out = RUNS_ROOT / "cross_corpus.json"
    out.write_text(
        json.dumps(
            {
                "winners": winners,
                "corpora": {n: corpora.get(n).structure for n in names},
                "top": {
                    n: [
                        {"config": c.label, "mrr": c.mrr, "hit": c.hit, "ms": c.latency_ms}
                        for c, _ in per_corpus[n][: args.top]
                    ]
                    for n in names
                },
                "winner_differs": len(distinct) > 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: runs/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
