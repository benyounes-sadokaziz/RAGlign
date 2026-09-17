"""The before/after validation study (Phase 2, design doc Part 3 point 4).

Scores every chunking strategy twice against the same questions and the same
retrievals:

  BEFORE  chunk-ID ground truth authored against one strategy's boundaries
  AFTER   span alignment, which no strategy authored

and reports whether the two disagree about which config is best.

The study is run once per author strategy. That matters: if the bias were an
artefact of picking a convenient author, rotating the author role would make it
disappear. It does not -- whoever authors, wins.

Run: .venv/Scripts/python.exe scripts/validation_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from raglign.alignment import align
from raglign.embedding import Embedder
from raglign.experiment import RUNS_ROOT, ConfigSpec, build_pipeline
from raglign.loader import corpus_fingerprint, index_by_id, load_documents
from raglign.qa import QA_ROOT, load_qa_set
from raglign.study import build_chunk_id_ground_truth, chunk_id_hits

SPECS = [
    ConfigSpec(chunker="fixed", chunker_params={"size": 800, "overlap": 100}),
    ConfigSpec(chunker="recursive", chunker_params={"size": 800, "overlap": 100}),
    ConfigSpec(chunker="heading", chunker_params={"max_size": 1200, "min_size": 200}),
    ConfigSpec(chunker="semantic", chunker_params={"percentile": 90.0, "max_size": 1200, "min_size": 200}),
]

K = 5
TAU = 0.5
MATCH_IOU = 0.9


def main(argv: list[str]) -> int:
    qa_name = argv[1] if len(argv) > 1 else "longspan_v1.json"
    docs = load_documents()
    qa = load_qa_set(QA_ROOT / qa_name, index_by_id(docs))
    if qa.rejections:
        print(f"refusing to run: {len(qa.rejections)} unresolved QA items")
        return 1

    print(f"corpus      : {len(docs)} docs, fingerprint {corpus_fingerprint(docs)}")
    print(f"qa set      : {qa_name}  ({len(qa)} questions)")
    print(f"scoring     : k={K}, span tau={TAU}, chunk-ID match IoU={MATCH_IOU}\n")

    embedder = Embedder()

    # Build every pipeline once and retrieve once; both scoring schemes read the
    # same retrievals, so any difference is attributable to the ground truth
    # alone and not to run-to-run variation.
    pipelines = {}
    retrievals = {}
    for spec in SPECS:
        print(f"  building {spec.id} ...", end=" ", flush=True)
        p = build_pipeline(spec, docs, embedder)
        pipelines[spec.id] = p
        retrievals[spec.id] = {
            q.id: [s.chunk for s in p.retriever.search(q.question, k=K)] for q in qa.items
        }
        print(f"{len(p.chunks)} chunks")
    print()

    report = {"corpus_fingerprint": corpus_fingerprint(docs), "qa_set": qa_name, "authors": {}}

    for author in SPECS:
        gold = build_chunk_id_ground_truth(qa.items, pipelines[author.id].chunks)
        print(f"AUTHOR STRATEGY: {author.id}")
        print("  (ground truth recorded as chunk IDs in this strategy's segmentation)\n")
        head = f"  {'evaluated config':<26} {'naive hit@5':>12} {'naive MRR':>10} {'span hit@5':>12} {'span MRR':>10} {'delta':>8}"
        print(head)
        print("  " + "-" * (len(head) - 2))

        rows = []
        for spec in SPECS:
            naive_h, naive_rr, span_h, span_rr = [], [], [], []
            for q in qa.items:
                chunks = retrievals[spec.id][q.id]

                nh = chunk_id_hits(chunks, gold[q.id], MATCH_IOU)
                naive_h.append(1.0 if any(nh) else 0.0)
                naive_rr.append(1.0 / (nh.index(True) + 1) if any(nh) else 0.0)

                a = align(q.id, chunks, q.spans)
                rank = a.first_hit_rank(TAU, K)
                span_h.append(1.0 if rank else 0.0)
                span_rr.append(1.0 / rank if rank else 0.0)

            n = len(qa.items)
            row = {
                "config": spec.id,
                "is_author": spec.id == author.id,
                "naive_hit": sum(naive_h) / n,
                "naive_mrr": sum(naive_rr) / n,
                "span_hit": sum(span_h) / n,
                "span_mrr": sum(span_rr) / n,
            }
            rows.append(row)
            mark = " *" if row["is_author"] else "  "
            print(
                f"  {spec.id:<26}{mark}{row['naive_hit']:>10.3f} {row['naive_mrr']:>10.3f} "
                f"{row['span_hit']:>12.3f} {row['span_mrr']:>10.3f} "
                f"{row['span_hit'] - row['naive_hit']:>+8.3f}"
            )

        naive_best = max(rows, key=lambda r: (r["naive_hit"], r["naive_mrr"]))["config"]
        span_best = max(rows, key=lambda r: (r["span_hit"], r["span_mrr"]))["config"]
        verdict = "DISAGREE" if naive_best != span_best else "agree"
        print(f"\n  winner under chunk-ID GT : {naive_best}")
        print(f"  winner under span GT     : {span_best}")
        print(f"  -> the two methodologies {verdict}")
        if naive_best == author.id and span_best != author.id:
            print("     chunk-ID GT crowned the strategy that authored it.")
        print()

        report["authors"][author.id] = {
            "rows": rows,
            "naive_winner": naive_best,
            "span_winner": span_best,
            "agree": naive_best == span_best,
        }

    # Headline: how much does merely having authored the ground truth pay?
    print("SELF-PREFERENCE (author's own naive score minus mean naive score of the others)")
    for author_id, data in report["authors"].items():
        own = next(r for r in data["rows"] if r["is_author"])
        others = [r for r in data["rows"] if not r["is_author"]]
        bonus_h = own["naive_hit"] - sum(r["naive_hit"] for r in others) / len(others)
        span_h = own["span_hit"] - sum(r["span_hit"] for r in others) / len(others)
        print(
            f"  {author_id:<28} chunk-ID GT: {bonus_h:>+7.3f}   span GT: {span_h:>+7.3f}"
        )
        data["self_preference_naive"] = bonus_h
        data["self_preference_span"] = span_h

    # Is the effect an artefact of demanding near-identical chunks? Relax the
    # chunk-ID match all the way down to 0.3 IoU -- far more generous than
    # "this is the gold chunk" can honestly mean -- and check the bias survives.
    print("\nMATCH-STRICTNESS SENSITIVITY (self-preference in hit@5 under chunk-ID GT)")
    ious = [0.9, 0.7, 0.5, 0.3]
    print(f"  {'author strategy':<28}" + "".join(f"{'IoU>=' + str(i):>10}" for i in ious))
    sens = {}
    for author in SPECS:
        gold = build_chunk_id_ground_truth(qa.items, pipelines[author.id].chunks)
        row = []
        for iou in ious:
            scores = {}
            for spec in SPECS:
                hits = [
                    1.0 if any(chunk_id_hits(retrievals[spec.id][q.id], gold[q.id], iou)) else 0.0
                    for q in qa.items
                ]
                scores[spec.id] = sum(hits) / len(hits)
            others = [v for kk, v in scores.items() if kk != author.id]
            row.append(scores[author.id] - sum(others) / len(others))
        sens[author.id] = dict(zip(map(str, ious), row))
        print(f"  {author.id:<28}" + "".join(f"{v:>+10.3f}" for v in row))
    report["match_iou_sensitivity"] = sens

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    out = RUNS_ROOT / "validation_study.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: runs/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
