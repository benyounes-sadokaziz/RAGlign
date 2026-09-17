"""RAGlign demo UI.

Reads saved run manifests only -- it never evaluates anything itself. Keeping
the demo a pure view over runs/ means the slider is instant, the numbers on
screen are exactly the numbers committed to disk, and nothing shown here can
have been computed differently from the CLI results.

Run: .venv/Scripts/streamlit.exe run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from raglign.optimizer import RUNS_ROOT, Weights, explain, load_candidates, pareto_frontier, rank

st.set_page_config(page_title="RAGlign", layout="wide")

QA_SETS = {
    "longspan_v1.json": "Long-span (11 q, evidence 320-533 chars)",
    "handwritten_v1.json": "Short-span (24 q, evidence ~123 chars)",
}


@st.cache_data
def _load(qa_file: str):
    return load_candidates(qa_file)


st.title("RAGlign")
st.caption(
    "Evaluation-driven RAG pipeline optimization with chunking-independent ground truth"
)

with st.sidebar:
    st.header("Evaluation set")
    qa_file = st.selectbox(
        "Question set", list(QA_SETS), format_func=lambda k: QA_SETS[k], label_visibility="collapsed"
    )
    st.header("Priorities")
    wq = st.slider("Quality", 0.0, 1.0, 0.70, 0.05)
    wl = st.slider("Latency", 0.0, 1.0, 0.20, 0.05)
    wc = st.slider("Context size", 0.0, 1.0, 0.10, 0.05)
    total = max(wq + wl + wc, 1e-9)
    weights = Weights(wq / total, wl / total, wc / total)
    st.caption(f"normalised: {weights.quality:.0%} / {weights.latency:.0%} / {weights.context:.0%}")

try:
    cands = _load(qa_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if not cands:
    st.warning(f"No runs found for {qa_file}. Run `scripts/run_grid.py {qa_file}` first.")
    st.stop()

ranked = rank(cands, weights)
frontier = {c.config_id for c in pareto_frontier(cands)}
winner = ranked[0][0]

tab_rec, tab_grid, tab_study = st.tabs(
    ["Recommendation", "Full grid", "Validation study"]
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

    st.subheader("Pareto frontier")
    st.caption(
        "Nothing here is strictly worse than anything else on quality, latency and context "
        "simultaneously. Everything off the frontier is dominated."
    )
    fdf = pd.DataFrame(
        [
            {
                "config": c.label,
                "quality": round(c.quality, 3),
                "latency_ms": round(c.latency_ms, 1),
                "chars": round(c.chars),
            }
            for c in sorted(pareto_frontier(cands), key=lambda c: -c.quality)
        ]
    )
    st.dataframe(fdf, use_container_width=True, hide_index=True)
    st.scatter_chart(fdf, x="latency_ms", y="quality", size="chars")

with tab_grid:
    df = pd.DataFrame(
        [
            {
                "rank": i,
                "config": c.label,
                "score": round(s, 3),
                "MRR": round(c.mrr, 3),
                "hit@5": round(c.hit, 3),
                "nDCG": round(c.ndcg, 3),
                "soft": round(c.soft, 3),
                "ms/query": round(c.latency_ms, 1),
                "chars": round(c.chars),
                "chunks": c.n_chunks,
                "pareto": "yes" if c.config_id in frontier else "",
            }
            for i, (c, s) in enumerate(rank(cands, weights), 1)
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True, height=560)

    st.subheader("Segmentation ceiling (oracle reachability)")
    st.caption(
        "Fraction of evidence spans that *some* chunk could satisfy, ignoring the retriever "
        "entirely. Below 1.0 means the chunker destroyed evidence before retrieval began -- a "
        "failure hit@k reports as a retrieval problem."
    )
    seen, rows = set(), []
    for c in cands:
        key = c.label.split("/")[0]
        if key in seen:
            continue
        seen.add(key)
        rows.append({"chunker": key, **{f"tau={t}": round(float(v), 3) for t, v in sorted(c.oracle.items())}})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_study:
    study_path = RUNS_ROOT / "validation_study.json"
    if not study_path.exists():
        st.warning("Run `scripts/validation_study.py` first.")
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
                sdf = pd.DataFrame(
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
                )
                st.dataframe(sdf, use_container_width=True, hide_index=True)
                st.write(
                    f"chunk-ID winner: **{data['naive_winner']}** | "
                    f"span winner: **{data['span_winner']}**"
                )

        st.subheader("Self-preference: what authoring the ground truth is worth")
        sp = pd.DataFrame(
            [
                {
                    "author strategy": a,
                    "chunk-ID GT": round(d["self_preference_naive"], 3),
                    "span GT": round(d["self_preference_span"], 3),
                }
                for a, d in study["authors"].items()
            ]
        )
        st.dataframe(sp, use_container_width=True, hide_index=True)
        st.caption(
            "Author's own hit@5 minus the mean of the others. Under chunk-ID ground truth the "
            "author gains ~0.6 regardless of real quality; under span alignment the spread "
            "collapses to the genuine differences."
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
