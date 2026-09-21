"""RAGlign dashboard.

A read-only view over saved run manifests. It never evaluates anything: the
controls re-rank pre-computed results, so interaction is instant and every
number on screen is identical to the one the CLI wrote to disk. Nothing shown
here can have been computed a second, different way.

Layout and styling live in `ui/`; this module only decides what goes where.

Run: .venv/Scripts/streamlit.exe run app.py
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from raglign import corpora
from raglign.diagnosis import Cause, diagnose_config, recommended_action
from raglign.optimizer import explain
from ui import charts, data
from ui.components import (
    badge,
    field_label,
    picker_card,
    scatter_legend,
    section_header,
    weight_row,
    card,
    esc,
    fmt,
    hint,
    kpi,
    kpi_row,
    legend,
    note,
    page_head,
    pm,
    prose,
    sidebar_brand,
    sidebar_title,
    spec_line,
    strip,
    table,
    topbar,
)
from ui.theme import FAMILY_COLORS, css, family_of

st.set_page_config(page_title="RAGlign", page_icon="◈", layout="wide")
st.markdown(css(), unsafe_allow_html=True)

VIEWS = {
    "Recommendation": "⭐",
    "Full grid": "▦",
    "Failure diagnosis": "⚠",
    "Validation study": "⚖",
    "Cross-corpus": "🗄",
}

CAUSE_LABELS = {
    Cause.CHUNKER_DESTROYED: "Chunker destroyed",
    Cause.CHUNKER_DEGRADED: "Chunker degraded",
    Cause.RERANKER_DEMOTED: "Reranker demoted",
    Cause.RETRIEVER_MISSED: "Retriever missed",
    Cause.RANKED_LOW: "Ranked low",
    Cause.OK: "Answered at rank 1",
}


# --------------------------------------------------------------------------- sidebar
CORPUS_BLURB = {
    "fastapi": "API docs: headings, code blocks, lists.",
    "prose": "Flowing prose: no markup, no headings.",
}

with st.sidebar:
    st.markdown(sidebar_brand(), unsafe_allow_html=True)
    page = st.radio(
        "view",
        list(VIEWS),
        format_func=lambda v: f"{VIEWS[v]}  {v}",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(section_header("Global controls"), unsafe_allow_html=True)

    corpus_meta = {name: (structure, n) for name, structure, n in data.available_corpora()}
    st.markdown(field_label("Corpus"), unsafe_allow_html=True)
    corpus = st.selectbox(
        "Corpus",
        list(corpus_meta),
        format_func=lambda c: f"{c} ({corpus_meta[c][1]} docs)",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="rg-sel-note" style="font-size:.7rem;color:#8fa0bd;margin:-.35rem .2rem .2rem">'
        f"{esc(CORPUS_BLURB.get(corpus, corpus_meta[corpus][0]))}</div>",
        unsafe_allow_html=True,
    )

    sets = corpora.get(corpus).qa_sets()
    if not sets:
        st.error(f"No QA sets for {corpus}")
        st.stop()
    st.markdown(field_label("Question set"), unsafe_allow_html=True)
    qa_file = st.selectbox(
        "Question set",
        sets,
        format_func=lambda s: "Long-span questions" if "longspan" in s else "Short-span questions",
        label_visibility="collapsed",
    )

    st.markdown(field_label("Overlap threshold τ"), unsafe_allow_html=True)
    tau = st.segmented_control(
        "tau",
        options=[0.1, 0.3, 0.5, 0.7],
        default=0.5,
        format_func=lambda v: f"{v:.1f}",
        label_visibility="collapsed",
    )
    if tau is None:
        tau = 0.5

    st.markdown(
        section_header("Priority weights", "ⓘ").replace("rg-sec", "rg-sec rg-sec-sm"),
        unsafe_allow_html=True,
    )

    # Each slider is preceded by its own name/value row, so the chip sits to the
    # right of the label as in the design rather than floating over the track.
    st.markdown('<div class="rg-s-quality">', unsafe_allow_html=True)
    st.markdown(weight_row("Quality", st.session_state.get("wq", 0.70)), unsafe_allow_html=True)
    wq = st.slider("Quality", 0.0, 1.0, 0.70, 0.05, key="wq", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rg-s-latency">', unsafe_allow_html=True)
    st.markdown(weight_row("Latency", st.session_state.get("wl", 0.20)), unsafe_allow_html=True)
    wl = st.slider("Latency", 0.0, 1.0, 0.20, 0.05, key="wl", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rg-s-context">', unsafe_allow_html=True)
    st.markdown(weight_row("Context", st.session_state.get("wc", 0.10)), unsafe_allow_html=True)
    wc = st.slider("Context", 0.0, 1.0, 0.10, 0.05, key="wc", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        hint(
            "Weights re-rank all configurations live. Values are normalised (0.0 – 1.0) "
            "and change the ranking only — never the measurements."
        ),
        unsafe_allow_html=True,
    )

try:
    view = data.build(corpus, qa_file, tau, wq, wl, wc)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if view is None:
    st.markdown(topbar("No results for this selection"), unsafe_allow_html=True)
    st.warning(
        f"No saved runs for **{corpus} / {qa_file}**. "
        f"Generate them with `scripts/run_grid.py --corpus {corpus}`."
    )
    st.stop()

st.markdown(
    topbar(f"Results loaded (read-only) · {len(view.candidates)} configs · τ={tau}"),
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- views
def render_recommendation() -> None:
    w, score = view.winner, view.winner_score
    ci = view.intervals[w.config_id]
    avg_mrr = data.field_average(view, "mrr")
    avg_hit = data.field_average(view, "hit")
    avg_lat = data.field_average(view, "latency_ms")
    avg_chars = data.field_average(view, "chars")

    st.markdown(
        page_head(
            "⭐",
            "Recommendation",
            "The best overall configuration for your current settings, with key metrics, "
            "explanation and actionable insight.",
        ),
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1.35, 0.95, 1.05], gap="medium")

    with left:
        tiles = kpi_row(
            [
                kpi("MRR@5", fmt(w.mrr), (ci["mrr"][1], ci["mrr"][2]),
                    (w.mrr - avg_mrr) / avg_mrr if avg_mrr else None),
                kpi("Hit@5", fmt(w.hit), (ci["hit"][1], ci["hit"][2]),
                    (w.hit - avg_hit) / avg_hit if avg_hit else None),
                kpi("Latency (ms/query)", f"{w.latency_ms:,.0f}", None,
                    (w.latency_ms - avg_lat) / avg_lat if avg_lat else None,
                    lower_is_better=True),
                kpi("Context size (chars)", f"{w.chars:,.0f}", None,
                    (w.chars - avg_chars) / avg_chars if avg_chars else None,
                    lower_is_better=True),
            ]
        )
        st.markdown(
            card(badge("Recommended Configuration") + spec_line(data.spec_pairs(w)) + tiles),
            unsafe_allow_html=True,
        )

    with mid:
        body = prose(esc(explain(w, view.candidates, view.weights)).replace("\n", "<br>"))
        action = ""
        if w.per_question:
            base = view.baselines.get((w.chunker, w.retriever)) if w.reranker else None
            fix = recommended_action(diagnose_config(w.per_question, base))
            if fix:
                action = note("Highest-leverage fix", esc(fix))
        st.markdown(card(body + action, "Why this works"), unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="rg-card"><h3>Quality vs Latency <span class="rg-mute">&#9432;</span></h3>'
            '<p class="rg-sub">Log latency axis; point size is context cost.</p></div>',
            unsafe_allow_html=True,
        )
        df, front = data.scatter_frame(view)
        plot, leg = st.columns([2.1, 1.0], gap="small")
        with plot:
            st.altair_chart(charts.quality_latency(df, front), use_container_width=True)
        with leg:
            fams = list(dict.fromkeys(df["family"]))
            lo, hi = df["chars"].min(), df["chars"].max()
            steps = [lo + (hi - lo) * f for f in (0, 0.33, 0.66, 1.0)]
            st.markdown(
                scatter_legend(fams, [f"{v/1000:.1f}K" if v >= 1000 else f"{v:.0f}" for v in steps]),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    lower, side = st.columns([1.6, 1.0], gap="medium")

    with lower:
        front_cands = [c for c, _ in view.ranked if c.config_id in view.frontier_ids]
        rows = []
        for i, c in enumerate(front_cands, 1):
            iv = view.intervals[c.config_id]
            mark = " 🏆" if c.config_id == w.config_id else ""
            rows.append(
                [
                    f'<span class="rg-rank">{i}</span>',
                    f"{esc(c.label)}{mark}",
                    pm(c.mrr, (iv["mrr"][2] - iv["mrr"][1]) / 2),
                    pm(c.hit, (iv["hit"][2] - iv["hit"][1]) / 2),
                    f"{c.latency_ms:,.1f}",
                    f"{c.chars:,.0f}",
                ]
            )
        winner_idx = next(
            (i for i, c in enumerate(front_cands) if c.config_id == w.config_id), None
        )
        st.markdown(
            card(
                table(
                    ["#", "Config", "MRR@5 ↑", "Hit@5 ↑", "Latency (ms) ↓", "Context (chars) ↓"],
                    rows,
                    winner_row=winner_idx,
                ),
                "Pareto frontier",
                "Non-dominated configurations — nothing here is beaten on every axis at once.",
            ),
            unsafe_allow_html=True,
        )

    with side:
        fam = data.family_frame(view)
        total = int(fam["count"].sum())
        st.markdown(
            '<div class="rg-card"><h3>Config distribution '
            '<span class="rg-mute">(current settings)</span></h3></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.altair_chart(charts.family_donut(fam, total), use_container_width=True)
        with c2:
            st.markdown(
                legend(
                    [
                        (row.family, FAMILY_COLORS.get(row.family, "#3b82f6"), int(row.count),
                         row.count / total)
                        for row in fam.itertuples()
                    ]
                ),
                unsafe_allow_html=True,
            )

        top_rows = [
            [f'<span class="rg-rank">{i}</span>', esc(c.label), f"{s:.3f}"]
            for i, (c, s) in enumerate(view.ranked[:5], 1)
        ]
        st.markdown(
            card(table(["#", "Config", "Score"], top_rows, numeric_from=2), "Top 5 by composite score"),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    runner_up = view.ranked[1][0] if len(view.ranked) > 1 else None
    if runner_up:
        d_mrr = w.mrr - runner_up.mrr
        d_lat = (runner_up.latency_ms - w.latency_ms) / max(w.latency_ms, 0.01)
        insight = (
            f'The top configuration leads on the weighted score. The runner-up '
            f'<span class="rg-hl">{esc(runner_up.label)}</span> trades '
            f"<b>{abs(d_mrr):.3f} MRR</b> for <b>{d_lat:+.0%}</b> latency. "
        )
        iv_w, iv_r = view.intervals[w.config_id]["mrr"], view.intervals[runner_up.config_id]["mrr"]
        if not (iv_w[2] < iv_r[1] or iv_r[2] < iv_w[1]):
            insight += (
                "Their confidence intervals <b>overlap</b>, so at this sample size the two are "
                "not statistically separable — treat the ordering as provisional."
            )
        st.markdown(
            strip("&#9873;", "Key insight", insight, cta="View full grid"),
            unsafe_allow_html=True,
        )


def render_grid() -> None:
    st.markdown(
        page_head(
            "▦",
            "Full grid",
            "Every configuration at every recorded cutoff. Toggle metric families to keep the "
            "table readable.",
        ),
        unsafe_allow_html=True,
    )
    ks = view.winner.ks or [5]
    families = st.multiselect(
        "Metrics",
        ["hit@k", "recall@k", "MRR@k", "nDCG@k", "soft", "cost"],
        default=["hit@k", "MRR@k", "cost"],
        label_visibility="collapsed",
    )

    rows = []
    for i, (c, s) in enumerate(view.ranked, 1):
        row = {"#": i, "config": c.label, "score": round(s, 3)}
        for fam, metric, label in (
            ("hit@k", "hit_at_k", "hit"),
            ("recall@k", "recall_at_k", "rec"),
            ("MRR@k", "mrr", "MRR"),
            ("nDCG@k", "ndcg_at_k", "nDCG"),
        ):
            if fam in families:
                for k in ks:
                    v = c.metric_at(metric, k)
                    row[f"{label}@{k}"] = round(v, 3) if v is not None else None
        if "soft" in families:
            row["soft"] = round(c.soft, 3)
        if "cost" in families:
            row["ms/query"] = round(c.latency_ms, 1)
            row["chars"] = round(c.chars)
            row["chunks"] = c.n_chunks
        row["pareto"] = "★" if c.config_id in view.frontier_ids else ""
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=520)
    st.markdown(
        note(
            "How to read this",
            "Where <b>hit@k</b> climbs with k but <b>MRR@k</b> barely moves, the evidence is being "
            "retrieved but ranked poorly — that gap is what a reranker fixes, and it is invisible "
            "if you only look at hit@5.",
            icon="📘",
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    seen, ceiling = set(), []
    for c in view.candidates:
        key = c.label.split("/")[0]
        if key in seen:
            continue
        seen.add(key)
        ceiling.append({"chunker": key, **{f"τ={t}": round(float(v), 3) for t, v in sorted(c.oracle.items())}})
    st.markdown(
        card("", "Segmentation ceiling (oracle reachability)",
             "Share of evidence spans some chunk could satisfy, ignoring the retriever entirely. "
             "Below 1.000 means the chunker destroyed evidence before retrieval began."),
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(ceiling), use_container_width=True, hide_index=True)


def render_diagnosis() -> None:
    st.markdown(
        page_head(
            "⚠",
            "Failure diagnosis",
            "Which stage lost each question — and they have different fixes. Evidence destroyed by "
            "chunking cannot be recovered by any embedding model.",
        ),
        unsafe_allow_html=True,
    )
    totals = data.cause_totals(view)
    grand = sum(totals.values()) or 1

    cols = st.columns(len(CAUSE_LABELS))
    for col, (cause, label) in zip(cols, CAUSE_LABELS.items()):
        with col:
            n = totals.get(cause, 0)
            st.markdown(
                card(
                    f'<div class="rg-kpi-l">{esc(label)}</div>'
                    f'<div class="rg-kpi-v" style="font-size:1.45rem">{n}</div>'
                    f'<div class="rg-kpi-ci">{n / grand:.1%} of all cases</div>'
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    df = data.diagnosis_frame(view, limit=12)
    if df.empty:
        st.info("Re-run the grid to record per-question diagnostics.")
        return
    st.markdown(card("", "Cause mix by configuration", "Top 12 configurations."), unsafe_allow_html=True)
    st.altair_chart(charts.cause_stack(df), use_container_width=True)

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    w = view.winner
    base = view.baselines.get((w.chunker, w.retriever)) if w.reranker else None
    findings = diagnose_config(w.per_question, base)
    rows = [
        [esc(f.question_id), esc(CAUSE_LABELS.get(f.cause, f.cause.value)), esc(f.detail), esc(f.fix)]
        for f in findings
        if f.cause is not Cause.OK
    ]
    st.markdown(
        card(
            table(["Question", "Cause", "What happened", "Suggested fix"], rows, winner_row=None, numeric_from=99)
            if rows
            else prose("Every question was answered at rank 1 for this configuration."),
            f"Per-question detail — {w.label}",
            "Only the questions that did not land at rank 1.",
        ),
        unsafe_allow_html=True,
    )


def render_study() -> None:
    st.markdown(
        page_head(
            "⚖",
            "Validation study",
            "Does chunking-independent ground truth change which configuration wins?",
        ),
        unsafe_allow_html=True,
    )
    study = data.load_json(corpus, "validation_study.json")
    if not study:
        st.warning(f"Run `scripts/validation_study.py --corpus {corpus}` first.")
        return

    authors = study["authors"]
    disagree = [a for a, d in authors.items() if not d["agree"]]
    c1, c2, c3 = st.columns(3)
    for col, label, value, sub in (
        (c1, "Author rotations", str(len(authors)), "each strategy authors the ground truth once"),
        (c2, "Winner flipped", f"{len(disagree)} / {len(authors)}", "chunk-ID vs span alignment disagree"),
        (c3, "Max self-preference",
         f"{max(d['self_preference_naive'] for d in authors.values()):+.3f}",
         "hit@5 gained purely by having authored"),
    ):
        with col:
            st.markdown(
                card(
                    f'<div class="rg-kpi-l">{esc(label)}</div>'
                    f'<div class="rg-kpi-v" style="font-size:1.6rem">{esc(value)}</div>'
                    f'<div class="rg-kpi-ci">{esc(sub)}</div>'
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    rows = [
        [
            esc(a),
            f"{d['self_preference_naive']:+.3f}",
            f"{d['self_preference_span']:+.3f}",
            "DISAGREE" if not d["agree"] else "agree",
        ]
        for a, d in authors.items()
    ]
    st.markdown(
        card(
            table(["Ground truth authored by", "chunk-ID GT", "span GT", "Verdict"], rows, winner_row=None),
            "Self-preference",
            "The author's own hit@5 minus the mean of the others. Under chunk-ID ground truth the "
            "author gains regardless of real quality.",
        ),
        unsafe_allow_html=True,
    )

    if "match_iou_sensitivity" in study:
        st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
        sens = study["match_iou_sensitivity"]
        ious = sorted({k for d in sens.values() for k in d}, reverse=True)
        rows = [[esc(a)] + [f"{sens[a][i]:+.3f}" for i in ious] for a in sens]
        st.markdown(
            card(
                table(["Author"] + [f"IoU ≥ {i}" for i in ious], rows, winner_row=None),
                "Is it just strict matching?",
                "Relaxing what counts as 'the gold chunk' shrinks the bias monotonically — it "
                "converges on the span-GT column only once matching is loose enough to effectively "
                "be span overlap.",
            ),
            unsafe_allow_html=True,
        )

    for author, d in authors.items():
        with st.expander(
            f"Author: {author} — {'DISAGREE' if not d['agree'] else 'agree'}", expanded=not d["agree"]
        ):
            rows = [
                [
                    esc(r["config"]) + (" ★" if r["is_author"] else ""),
                    f"{r['naive_hit']:.3f}",
                    f"{r['span_hit']:.3f}",
                    f"{r['span_hit'] - r['naive_hit']:+.3f}",
                ]
                for r in d["rows"]
            ]
            st.markdown(
                table(["Config", "chunk-ID hit@5", "span hit@5", "Δ"], rows, winner_row=None),
                unsafe_allow_html=True,
            )


def render_cross() -> None:
    st.markdown(
        page_head(
            "🗄",
            "Cross-corpus",
            "Same 30 configurations, same methodology, structurally opposite corpora. Does the "
            "winner transfer?",
        ),
        unsafe_allow_html=True,
    )

    frames, winners = {}, {}
    for name in corpora.names():
        sets_ = [s for s in corpora.get(name).qa_sets() if "longspan" in s]
        if not sets_:
            continue
        other = data.build(name, sets_[0], tau, wq, wl, wc)
        if other is None:
            continue
        frames[name] = other
        winners[name] = other.winner.label

    if len(frames) < 2:
        st.info("Run the grid on a second corpus to populate this comparison.")
        return

    cols = st.columns(len(frames), gap="medium")
    for col, (name, v) in zip(cols, frames.items()):
        with col:
            rows = [
                [f'<span class="rg-rank">{i}</span>', esc(c.label), f"{c.mrr:.3f}", f"{c.hit:.3f}"]
                for i, (c, _) in enumerate(v.ranked[:5], 1)
            ]
            st.markdown(
                card(
                    table(["#", "Config", "MRR", "Hit@5"], rows),
                    f"{name} — top 5",
                    corpora.get(name).structure,
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    spreads = {}
    for name, v in frames.items():
        by_family: dict[str, list[float]] = {}
        for c in v.candidates:
            fam = c.label.split("/")[0]
            if fam.startswith("fixed300"):
                continue  # deliberate control, not a viable candidate
            by_family.setdefault(fam, []).append(c.mrr)
        means = [sum(x) / len(x) for x in by_family.values()]
        spreads[name] = (max(means) - min(means)) if means else 0.0

    rows = [[esc(n), f"{s:.3f}"] for n, s in spreads.items()]
    st.markdown(
        card(
            table(["Corpus", "MRR spread between viable chunkers"], rows, winner_row=None),
            "Does the chunking choice matter here?",
            "Averaged over every retriever and reranker setting, excluding the deliberately "
            "broken control.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rg-gap"></div>', unsafe_allow_html=True)
    if len(set(winners.values())) > 1:
        body = (
            "The winning configuration <b>differs by corpus</b> — config choice does not transfer "
            "across document structure, which is the condition that makes a per-corpus optimizer "
            "worth having."
        )
    else:
        only = next(iter(set(winners.values())))
        body = (
            f'The <b>same</b> configuration (<span class="rg-hl">{esc(only)}</span>) wins on every '
            "corpus. Reported as-is rather than buried: it weakens the case for optimizing the "
            "<i>winner</i> per corpus. The useful finding is the spread above — how much the "
            "chunking choice matters is strongly corpus-dependent."
        )
    st.markdown(strip("🧭", "Verdict", body), unsafe_allow_html=True)


{
    "Recommendation": render_recommendation,
    "Full grid": render_grid,
    "Failure diagnosis": render_diagnosis,
    "Validation study": render_study,
    "Cross-corpus": render_cross,
}[page]()
