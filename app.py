"""RAGlign demo UI.

Reads saved run manifests only -- it never evaluates anything itself. Keeping
the demo a pure view over runs/ means the sliders are instant, the numbers on
screen are exactly the numbers committed to disk, and nothing shown here can
have been computed differently from the CLI results.

Run: .venv/Scripts/streamlit.exe run app.py
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import streamlit as st

from raglign import corpora
from raglign.diagnosis import Cause, diagnose_config, recommended_action
from raglign.optimizer import (
    RUNS_ROOT,
    Weights,
    explain,
    load_candidates,
    pareto_frontier,
    rank,
)

st.set_page_config(page_title="RAGlign", layout="wide")


@st.cache_data
def _load(qa_file: str, corpus: str):
    return load_candidates(qa_file, corpus)


def _qa_label(name: str) -> str:
    return "Long-span (evidence ~ chunk size)" if "longspan" in name else "Short-span (saturated)"


st.title("RAGlign")
st.caption("Evaluation-driven RAG pipeline optimization with chunking-independent ground truth")

with st.sidebar:
    st.header("Corpus")
    corpus = st.selectbox("Corpus", corpora.names(), label_visibility="collapsed")
    c = corpora.get(corpus)
    st.caption(c.structure)

    sets = c.qa_sets()
    if not sets:
        st.error(f"no QA sets for {corpus}")
        st.stop()
    st.header("Evaluation set")
    qa_file = st.selectbox("Question set", sets, format_func=_qa_label, label_visibility="collapsed")

    st.header("Priorities")
    wq = st.slider("Quality", 0.0, 1.0, 0.70, 0.05)
    wl = st.slider("Latency", 0.0, 1.0, 0.20, 0.05)
    wc = st.slider("Context size", 0.0, 1.0, 0.10, 0.05)
    total = max(wq + wl + wc, 1e-9)
    weights = Weights(wq / total, wl / total, wc / total)
    st.caption(f"normalised: {weights.quality:.0%} / {weights.latency:.0%} / {weights.context:.0%}")

try:
    cands = _load(qa_file, corpus)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if not cands:
    st.warning(f"No runs for {corpus}/{qa_file}. Run `scripts/run_grid.py --corpus {corpus}` first.")
    st.stop()

ranked = rank(cands, weights)
frontier = {x.config_id for x in pareto_frontier(cands)}
winner = ranked[0][0]

# A config's non-reranked twin, needed to tell "the reranker demoted it" apart
# from "the first stage never had it".
baselines = {(x.chunker, x.retriever): x.per_question for x in cands if not x.reranker}

tab_rec, tab_grid, tab_diag, tab_study, tab_cross = st.tabs(
    ["Recommendation", "Full grid", "Failure diagnosis", "Validation study", "Cross-corpus"]
)

with tab_rec:
    left, right = st.columns([2, 3])
    with left:
        st.subheader(winner.label)
        a, b = st.columns(2)
        a.metric("MRR", f"{winner.mrr:.3f}")
        b.metric("hit@5", f"{winner.hit:.3f}")
        a.metric("Latency", f"{winner.latency_ms:.1f} ms")
        b.metric("Context", f"{winner.chars:.0f} chars")
    with right:
        st.code(explain(winner, cands, weights), language=None)
        if winner.per_question:
            action = recommended_action(
                diagnose_config(winner.per_question, baselines.get((winner.chunker, winner.retriever)) if winner.reranker else None)
            )
            if action:
                st.info(f"**Highest-leverage fix:** {action}")

    st.subheader("Pareto frontier")
    st.caption(
        "Nothing here is strictly worse than everything else on quality, latency and context "
        "simultaneously. Everything off the frontier is dominated."
    )
    fdf = pd.DataFrame(
        [
            {
                "config": x.label,
                "quality": round(x.quality, 3),
                "latency_ms": round(x.latency_ms, 1),
                "chars": round(x.chars),
            }
            for x in sorted(pareto_frontier(cands), key=lambda x: -x.quality)
        ]
    )
    st.dataframe(fdf, use_container_width=True, hide_index=True)
    st.scatter_chart(fdf, x="latency_ms", y="quality", size="chars")

with tab_grid:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "rank": i,
                    "config": x.label,
                    "score": round(s, 3),
                    "MRR": round(x.mrr, 3),
                    "hit@5": round(x.hit, 3),
                    "nDCG": round(x.ndcg, 3),
                    "soft": round(x.soft, 3),
                    "ms/query": round(x.latency_ms, 1),
                    "chars": round(x.chars),
                    "chunks": x.n_chunks,
                    "pareto": "yes" if x.config_id in frontier else "",
                }
                for i, (x, s) in enumerate(ranked, 1)
            ]
        ),
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    st.subheader("Segmentation ceiling (oracle reachability)")
    st.caption(
        "Fraction of evidence spans that *some* chunk could satisfy, ignoring the retriever "
        "entirely. Below 1.0 means the chunker destroyed evidence before retrieval began -- a "
        "failure hit@k reports as a retrieval problem."
    )
    seen, rows = set(), []
    for x in cands:
        key = x.label.split("/")[0]
        if key in seen:
            continue
        seen.add(key)
        rows.append({"chunker": key, **{f"tau={t}": round(float(v), 3) for t, v in sorted(x.oracle.items())}})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_diag:
    st.subheader("Which stage lost each question?")
    st.caption(
        "A metric says a question failed. This says which component to fix -- and they have "
        "different fixes. Evidence destroyed by chunking cannot be recovered by any embedding "
        "model. Only span-based ground truth can tell these apart."
    )

    causes = [Cause.CHUNKER_DESTROYED, Cause.CHUNKER_DEGRADED, Cause.RERANKER_DEMOTED,
              Cause.RETRIEVER_MISSED, Cause.RANKED_LOW, Cause.OK]
    fleet: Counter = Counter()
    rows = []
    for x, _ in ranked:
        if not x.per_question:
            continue
        base = baselines.get((x.chunker, x.retriever)) if x.reranker else None
        counts = Counter(f.cause for f in diagnose_config(x.per_question, base))
        fleet.update(counts)
        rows.append({"config": x.label, **{cz.value: counts.get(cz, 0) for cz in causes}})

    if not rows:
        st.info("Run the grid again to record per-question diagnostics.")
    else:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)

        total = sum(fleet.values()) or 1
        st.subheader("Across every config")
        st.dataframe(
            pd.DataFrame(
                [
                    {"cause": cz.value, "questions": fleet.get(cz, 0),
                     "share": f"{fleet.get(cz, 0) / total:.1%}"}
                    for cz in causes if fleet.get(cz, 0)
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(f"Per-question detail — {winner.label}")
        base = baselines.get((winner.chunker, winner.retriever)) if winner.reranker else None
        detail = [
            {"question": f.question_id, "cause": f.cause.value, "detail": f.detail, "fix": f.fix}
            for f in diagnose_config(winner.per_question, base)
        ]
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

with tab_study:
    study_path = RUNS_ROOT / corpus / "validation_study.json"
    if not study_path.exists():
        st.warning(f"Run `scripts/validation_study.py --corpus {corpus}` first.")
    else:
        study = json.loads(study_path.read_text(encoding="utf-8"))
        st.subheader("Does chunking-independent ground truth change the answer?")
        st.caption(
            "Each block authors the ground truth against one strategy's chunk boundaries, then "
            "scores every strategy twice: once by chunk-ID matching, once by span alignment. "
            "Same retrievals both times, so any difference is the ground truth alone."
        )

        for author, data in study["authors"].items():
            with st.expander(
                f"Author: {author} — {'DISAGREE' if not data['agree'] else 'agree'}",
                expanded=not data["agree"],
            ):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "config": r["config"],
                                "author": "*" if r["is_author"] else "",
                                "chunk-ID hit@5": round(r["naive_hit"], 3),
                                "span hit@5": round(r["span_hit"], 3),
                                "delta": round(r["span_hit"] - r["naive_hit"], 3),
                            }
                            for r in data["rows"]
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.write(
                    f"chunk-ID winner: **{data['naive_winner']}** | span winner: **{data['span_winner']}**"
                )

        st.subheader("Self-preference: what authoring the ground truth is worth")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "author strategy": a,
                        "chunk-ID GT": round(d["self_preference_naive"], 3),
                        "span GT": round(d["self_preference_span"], 3),
                    }
                    for a, d in study["authors"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Author's own hit@5 minus the mean of the others. Under chunk-ID ground truth the "
            "author gains regardless of real quality; under span alignment the spread collapses "
            "to the genuine differences."
        )

        if "match_iou_sensitivity" in study:
            st.subheader("Is it just strict matching?")
            st.dataframe(
                pd.DataFrame(
                    [{"author": a, **{f"IoU>={i}": round(v, 3) for i, v in d.items()}}
                     for a, d in study["match_iou_sensitivity"].items()]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Relaxing what counts as 'the gold chunk' shrinks the bias monotonically, "
                "converging on the span-GT values only once matching is loose enough to "
                "effectively *be* span overlap."
            )

with tab_cross:
    st.subheader("Does the best config transfer between corpora?")
    st.caption(
        "The premise of building an optimizer at all is that there is no universal best config. "
        "This is the test: same 30 configs, same methodology, structurally opposite corpora."
    )

    frames = []
    for name in corpora.names():
        sets_ = corpora.get(name).qa_sets()
        ls = [s for s in sets_ if "longspan" in s]
        if not ls:
            continue
        try:
            other = load_candidates(ls[0], name)
        except ValueError:
            continue
        if not other:
            continue
        top = rank(other, weights)
        frames.append(
            pd.DataFrame(
                [
                    {
                        "corpus": name,
                        "rank": i,
                        "config": x.label,
                        "MRR": round(x.mrr, 3),
                        "hit@5": round(x.hit, 3),
                    }
                    for i, (x, _) in enumerate(top[:5], 1)
                ]
            )
        )

    if len(frames) < 2:
        st.info("Run the grid on both corpora to populate this comparison.")
    else:
        combined = pd.concat(frames)
        st.dataframe(combined, use_container_width=True, hide_index=True)
        winners = {f["corpus"].iloc[0]: f["config"].iloc[0] for f in frames}
        st.write("**Top config per corpus:** " + " | ".join(f"`{k}` → {v}" for k, v in winners.items()))
        if len(set(winners.values())) > 1:
            st.success(
                "The winner differs by corpus. Config choice does not transfer — which is "
                "precisely why a per-corpus optimizer is worth having."
            )
        else:
            st.info(
                "The same config wins on both. That is a transferable default, and also a "
                "finding worth reporting."
            )
