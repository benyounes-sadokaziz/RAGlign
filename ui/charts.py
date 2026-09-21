"""Altair charts, styled to match the surrounding cards.

Altair rather than a new dependency: it ships with Streamlit, and it gives the
encoding control this design needs (colour by chunker family, size by context
cost, a separate line for the Pareto frontier) that `st.scatter_chart` does not.

One deliberate choice worth flagging: latency is drawn on a log axis. The
observed range runs from ~0.4 ms to ~3000 ms, so on a linear axis every
non-reranked configuration collapses into a single stripe against the y-axis and
the plot stops distinguishing exactly the comparisons a reader came to make. The
same reasoning already governs how the optimizer normalises latency for ranking.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from .theme import FAMILY_COLORS, TOKENS

_FONT = "Inter, Segoe UI, system-ui, sans-serif"


def _base(chart: alt.Chart) -> alt.Chart:
    # White background and explicit text colours: Altair otherwise picks up the
    # host theme, which rendered a black plot inside a white card when the
    # viewer's Streamlit was in dark mode.
    return chart.properties(background="white").configure_view(
        strokeWidth=0, fill="white"
    ).configure_axis(
        labelFont=_FONT,
        titleFont=_FONT,
        labelColor=TOKENS["ink_soft"],
        titleColor=TOKENS["ink_soft"],
        labelFontSize=10,
        titleFontSize=11,
        grid=True,
        gridColor="#eef2f7",
        domainColor="#e2e8f0",
        tickColor="#e2e8f0",
    ).configure_legend(
        labelFont=_FONT,
        titleFont=_FONT,
        labelColor=TOKENS["ink_soft"],
        titleColor=TOKENS["ink_soft"],
        labelFontSize=10,
        titleFontSize=10,
        symbolStrokeWidth=0,
    )


def quality_latency(df: pd.DataFrame, frontier: pd.DataFrame, height: int = 260) -> alt.LayerChart:
    """Quality against latency, coloured by chunker family, sized by context cost.

    Three variables on one plot because the recommendation is a three-way
    trade-off; showing quality alone would hide the cost of buying it.
    """
    domain = [k for k in FAMILY_COLORS if k != "other"] + ["other"]
    scale = alt.Scale(domain=domain, range=[FAMILY_COLORS[k] for k in domain])

    # Reranked configs are drawn filled, non-reranked hollow, matching the legend.
    # Encoding the reranker as fill rather than another colour keeps colour meaning
    # exactly one thing (chunker family) across the whole dashboard.
    points = (
        alt.Chart(df)
        .mark_point(opacity=0.88, strokeWidth=1.6, size=90)
        .encode(
            x=alt.X(
                "latency_ms:Q",
                scale=alt.Scale(type="log"),
                title="Latency (ms/query)",
            ),
            y=alt.Y("quality:Q", scale=alt.Scale(domain=[0, 1]), title="Quality"),
            color=alt.Color("family:N", scale=scale, title="Chunker", legend=None),
            fill=alt.condition(
                alt.datum.reranked,
                alt.Color("family:N", scale=scale, legend=None),
                alt.value("transparent"),
            ),
            size=alt.Size("chars:Q", scale=alt.Scale(range=[40, 340]), legend=None),
            tooltip=[
                alt.Tooltip("config:N", title="Config"),
                alt.Tooltip("quality:Q", format=".3f"),
                alt.Tooltip("mrr:Q", title="MRR", format=".3f"),
                alt.Tooltip("latency_ms:Q", title="ms/query", format=".1f"),
                alt.Tooltip("chars:Q", title="chars", format=".0f"),
                alt.Tooltip("reranked:N", title="Reranker"),
            ],
        )
    )

    line = (
        alt.Chart(frontier)
        .mark_line(color=TOKENS["blue"], strokeWidth=2.2, point=False, opacity=0.9)
        .encode(
            x=alt.X("latency_ms:Q", scale=alt.Scale(type="log")),
            y=alt.Y("quality:Q", scale=alt.Scale(domain=[0, 1])),
        )
    )

    return _base((line + points).properties(height=height))


def family_donut(df: pd.DataFrame, total: int, height: int = 215) -> alt.LayerChart:
    """Share of evaluated configurations by chunker family, with the count inside."""
    domain = list(df["family"])
    scale = alt.Scale(domain=domain, range=[FAMILY_COLORS.get(f, "#3b82f6") for f in domain])

    ring = (
        alt.Chart(df)
        .mark_arc(innerRadius=58, outerRadius=88, stroke="white", strokeWidth=2)
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color("family:N", scale=scale, legend=None),
            tooltip=[
                alt.Tooltip("family:N", title="Chunker"),
                alt.Tooltip("count:Q", title="Configs"),
            ],
        )
    )
    centre = (
        alt.Chart(pd.DataFrame({"n": [total]}))
        .mark_text(fontSize=26, fontWeight=700, color=TOKENS["ink"], dy=-6, font=_FONT)
        .encode(text="n:Q")
    )
    caption = (
        alt.Chart(pd.DataFrame({"t": ["configurations"]}))
        .mark_text(fontSize=10, color=TOKENS["ink_soft"], dy=14, font=_FONT)
        .encode(text="t:N")
    )
    return _base((ring + centre + caption).properties(height=height))


def metric_bars(df: pd.DataFrame, value: str, title: str, height: int = 270) -> alt.Chart:
    """Horizontal bars for a single metric, used where a table would be harder to scan."""
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color=TOKENS["blue"], opacity=0.85)
        .encode(
            x=alt.X(f"{value}:Q", title=title),
            y=alt.Y("config:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("config:N"), alt.Tooltip(f"{value}:Q", format=".3f")],
        )
    )
    return _base(chart.properties(height=height))


def cause_stack(df: pd.DataFrame, height: int = 300) -> alt.Chart:
    """Failure causes per config as a stacked share.

    Colour carries meaning here rather than decoration: chunker-stage causes are
    warm (a fix that no retriever change can substitute for), retrieval-stage
    causes cool, resolved cases green.
    """
    order = [
        "chunker_destroyed",
        "chunker_degraded",
        "reranker_demoted",
        "retriever_missed",
        "ranked_low",
        "ok",
    ]
    colours = ["#b91c1c", TOKENS["amber"], TOKENS["violet"], "#60a5fa", "#cbd5e1", TOKENS["green"]]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", stack="normalize", title="Share of questions", axis=alt.Axis(format="%")),
            y=alt.Y("config:N", sort=None, title=None),
            color=alt.Color(
                "cause:N",
                scale=alt.Scale(domain=order, range=colours),
                title="Cause",
                sort=order,
            ),
            order=alt.Order("order:Q"),
            tooltip=["config:N", "cause:N", "count:Q"],
        )
    )
    return _base(chart.properties(height=height))
