"""Generation-side evaluation for a handful of configs.

Answers the question the retrieval metrics cannot: does better retrieval actually
produce better answers?

Deliberately NOT run over the whole grid. 30 configs x ~42 questions x (1
generation + 2 judgements x repeats) is thousands of API calls to answer a
question that a spread of five configs answers just as well. Configs are sampled
across the quality range -- best, worst, and evenly spaced between -- because a
correlation needs variation in the predictor, and the top five configs are all
clustered at the top.

Two generators:

  --generator mistral     an LLM answers from the retrieved context (needs a key
                          with quota); judged metrics are available on top.
  --generator extractive  the top-ranked chunk IS the answer. No API, no cost,
                          runs offline. A baseline rather than a strategy: it
                          measures how much of the answer retrieval already put
                          in first place, which is the floor any generator must
                          beat, since a reader cannot state what was never
                          retrieved.

The deterministic metrics (refusal rate, token F1 against the authored answer,
grounding against the ground-truth evidence span) apply to both. They are only
computable at all because this project has span-level ground truth.

Run: .venv/Scripts/python.exe scripts/run_generation.py --corpus fastapi --generator extractive
"""

from __future__ import annotations

import json

from _cli import parser, resolve_qa

from raglign import corpora, mistral
from raglign.embedding import Embedder
from raglign.experiment import RUNS_ROOT, ConfigSpec, build_pipeline
from raglign.generation import (
    correlation,
    evaluate_generation,
    extractive_answerer,
    llm_answerer,
)
from raglign.loader import index_by_id, load_corpus
from raglign.optimizer import load_candidates
from raglign.qa import load_qa_set, qa_path

K = 5


def _spec_from_label(cand) -> ConfigSpec:
    """Rebuild the ConfigSpec for a saved candidate so it can be re-run."""
    size = int("".join(ch for ch in cand.label.split("/")[0] if ch.isdigit()))
    if cand.chunker == "fixed":
        params = {"size": size, "overlap": 100 if size >= 800 else 50}
    elif cand.chunker == "recursive":
        params = {"size": size, "overlap": 100}
    elif cand.chunker == "heading":
        params = {"max_size": size, "min_size": 200}
    else:
        params = {"percentile": 90.0, "max_size": size, "min_size": 200}
    return ConfigSpec(
        chunker=cand.chunker,
        chunker_params=params,
        retriever=cand.retriever,
        reranker=cand.reranker,
    )


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--configs", type=int, default=5, help="how many configs to sample")
    p.add_argument("--repeats", type=int, default=3, help="judge repeats per question")
    p.add_argument("--questions", type=int, default=0, help="cap questions (0 = all)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument(
        "--generator",
        choices=["mistral", "extractive"],
        default="mistral" if mistral.available() else "extractive",
        help="how answers are produced (default: mistral when a key is present)",
    )
    args = p.parse_args()

    qa_file = resolve_qa(args.corpus, args.qa)
    docs = load_corpus(args.corpus)
    by_id = index_by_id(docs)
    qa = load_qa_set(qa_path(args.corpus, qa_file), by_id)
    if qa.rejections:
        print(f"refusing to run: {len(qa.rejections)} unresolved QA items")
        return 1

    items = qa.items[: args.questions] if args.questions else qa.items
    evidence = {
        it.id: " ".join(by_id[s.doc_id].text[s.start : s.end] for s in it.spans)
        for it in items
    }

    cands = load_candidates(qa_file, args.corpus)
    if not cands:
        print(f"no runs for {args.corpus}/{qa_file}; run scripts/run_grid.py first")
        return 1

    # Sample across the MRR range rather than taking the top N: a correlation
    # needs spread in the predictor, and the best configs are all bunched together.
    ordered = sorted(cands, key=lambda c: -c.mrr)
    n = min(args.configs, len(ordered))
    picks = [ordered[round(i * (len(ordered) - 1) / max(n - 1, 1))] for i in range(n)]

    use_llm = args.generator == "mistral"
    use_judge = use_llm and not args.no_judge and mistral.available()
    client = None
    if use_llm or use_judge:
        if not mistral.available():
            print("MISTRAL_API_KEY not set; use --generator extractive to run offline.")
            return 1
        client = mistral.MistralClient()

    print(f"corpus   : {args.corpus} -- {corpora.get(args.corpus).structure}")
    print(f"qa set   : {qa_file} ({len(items)} questions)")
    print(f"configs  : {n} sampled across the MRR range")
    print(f"generator: {'mistral ' + client.model if use_llm else 'extractive (top-1 chunk, no LLM)'}")
    print(f"judge    : {'mistral ' + client.model if use_judge else 'OFF'}")
    if use_judge:
        print(f"repeats  : {args.repeats}")
    print()

    embedder = Embedder()
    summaries = []
    for cand in picks:
        spec = _spec_from_label(cand)
        pipeline = build_pipeline(spec, docs, embedder)
        retrievals = {
            it.id: [s.chunk for s in pipeline.retriever.search(it.question, k=K)]
            for it in items
        }

        answerer = (
            llm_answerer(
                lambda prompt, system=None: client.chat(
                    prompt, system=system, temperature=0.0, max_tokens=300
                )
            )
            if use_llm
            else extractive_answerer()
        )

        print(f"  {cand.label:<28} mrr={cand.mrr:.3f} ...", end=" ", flush=True)
        summary = evaluate_generation(
            cand.config_id,
            items,
            retrievals,
            evidence,
            answerer=answerer,
            chat_json=(client.chat_json if use_judge else None),
            repeats=args.repeats,
        )
        summary_dict = summary.as_dict()
        summary_dict["retrieval_mrr"] = cand.mrr
        summary_dict["retrieval_hit"] = cand.hit
        summary_dict["label"] = cand.label
        summaries.append(summary_dict)
        print(
            f"refusal={summary.refusal_rate:.2f} goldF1={summary.gold_f1:.3f} "
            f"ground={summary.span_grounding:.3f}"
            + (f" faith={summary.faithfulness:.3f}" if summary.faithfulness is not None else "")
        )

    print()
    header = (
        f"{'config':<28}{'ret.MRR':>9}{'refusal':>9}{'goldF1':>9}{'goldRec':>9}"
        f"{'ground':>9}{'faith':>9}{'relev':>9}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        f = f"{s['faithfulness']:>9.3f}" if s["faithfulness"] is not None else f"{'-':>9}"
        r = f"{s['relevance']:>9.3f}" if s["relevance"] is not None else f"{'-':>9}"
        print(
            f"{s['label']:<28}{s['retrieval_mrr']:>9.3f}{s['refusal_rate']:>9.2f}"
            f"{s['gold_f1']:>9.3f}{s['gold_recall']:>9.3f}"
            f"{s['span_grounding']:>9.3f}{f}{r}"
        )

    print("\nDOES RETRIEVAL QUALITY PREDICT ANSWER QUALITY?")
    mrrs = [s["retrieval_mrr"] for s in summaries]
    for name, key in (("gold F1", "gold_f1"), ("gold recall", "gold_recall"),
                      ("grounding", "span_grounding"),
                      ("faithfulness", "faithfulness"), ("relevance", "relevance")):
        ys = [s[key] for s in summaries]
        if any(y is None for y in ys):
            print(f"  retrieval MRR vs {name:<14} not judged")
            continue
        r = correlation(mrrs, ys)
        print(f"  retrieval MRR vs {name:<14} r = {r:+.3f}" if r is not None
              else f"  retrieval MRR vs {name:<14} undefined (constant series)")
    if not use_llm:
        print(
            "\n  NOTE: with the extractive baseline the whole chunk is the answer, so gold F1\n"
            "  is confounded by chunk size -- its precision term penalises length, which is\n"
            "  why it can come out negative here. Read gold RECALL and grounding instead;\n"
            "  F1 only becomes meaningful once a real generator produces short answers."
        )
    print(
        "\n  A strong positive r is the justification for optimizing retrieval at all.\n"
        "  A weak one would mean the retrieval metrics, however precise, are not\n"
        "  tracking what reaches the user -- worth reporting either way."
    )

    out_dir = RUNS_ROOT / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "generation.json"
    payload = {
        "corpus": args.corpus,
        "qa_file": qa_file,
        "n_questions": len(items),
        "generator": client.model if use_llm else "extractive",
        "judge": client.model if use_judge else None,
        "repeats": args.repeats if use_judge else 0,
        "usage": client.usage.as_dict() if client else None,
        "configs": summaries,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: runs/{args.corpus}/{out.name}")
    if client:
        print(f"usage  : {client.usage.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
