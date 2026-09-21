"""Presentational pieces: cards, KPI tiles, callouts, tables.

Each returns an HTML string rather than writing to Streamlit directly, so a
caller can compose them inside one `st.markdown` block. That matters: rendering
several adjacent `st.markdown` calls produces separate wrapper divs that break
the card grid, and returning strings keeps layout decisions with the layout code.

Every value arriving here is already computed. Nothing in this module reads run
data or decides what is "best" -- presentation stays separate from the scoring,
so a change to how a number looks can never change the number.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

from .theme import FAMILY_COLORS, FAMILY_LABELS, TOKENS


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt(value: float | None, prec: int = 3, dash: str = "—") -> str:
    """Numbers only; None renders as a dash rather than a zero.

    The distinction is load-bearing throughout this project: a missing judgement
    and a score of zero mean opposite things, and a table that renders both as
    0.000 invites the wrong conclusion.
    """
    return dash if value is None else f"{value:.{prec}f}"


def page_head(icon: str, title: str, subtitle: str) -> str:
    return (
        f'<div class="rg-head"><div class="rg-head-icon">{icon}</div>'
        f"<div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div></div>"
    )


def topbar(status: str) -> str:
    return (
        f'<div class="rg-topbar"><span class="rg-pill">'
        f'<span class="rg-dot"></span>{esc(status)}</span>'
        f'<span class="rg-moon">&#9789;</span></div>'
    )


def card(body: str, title: str | None = None, subtitle: str | None = None) -> str:
    head = ""
    if title:
        head += f"<h3>{esc(title)}</h3>"
    if subtitle:
        head += f'<p class="rg-sub">{esc(subtitle)}</p>'
    return f'<div class="rg-card">{head}{body}</div>'


def badge(text: str, icon: str = "&#127942;") -> str:
    return f'<div class="rg-badge">{icon} {esc(text)}</div>'


def spec_line(pairs: Sequence[tuple[str, str]]) -> str:
    """'Chunking: Heading (1200) • Retrieval: Hybrid • Reranker: ON'"""
    parts = [f'<span class="rg-k">{esc(k)}:</span> <b>{esc(v)}</b>' for k, v in pairs]
    return '<div class="rg-spec">' + '<span class="rg-sep">&bull;</span>'.join(parts) + "</div>"


def picker_card(icon: str, title: str, detail: str) -> str:
    """The sidebar selector's visible face: icon, bold title, muted description."""
    return (
        f'<div class="rg-pick"><div class="rg-pick-ico">{icon}</div>'
        f'<div><div class="rg-pick-t">{esc(title)}</div>'
        f'<div class="rg-pick-d">{esc(detail)}</div></div>'
        f'<div class="rg-pick-ch">&#9662;</div></div>'
    )


def section_header(text: str, chevron: str = "&#9652;") -> str:
    return f'<div class="rg-sec">{esc(text)}<span class="chev">{chevron}</span></div>'


def field_label(text: str) -> str:
    return f'<div class="rg-lab">{esc(text)}</div>'


def weight_row(name: str, value: float) -> str:
    return (
        f'<div class="rg-wrow"><span class="n">{esc(name)}</span>'
        f'<span class="rg-chip">{value:.1f}</span></div>'
    )


def scatter_legend(families: Sequence[str], sizes: Sequence[str]) -> str:
    """Legend matching the design: frontier line, families, reranker, context size.

    Written by hand rather than left to Altair because the design groups four
    distinct encodings under their own headings, which a stacked automatic legend
    cannot express.
    """
    fam_rows = "".join(
        f'<div class="r"><span class="sw" style="background:{FAMILY_COLORS.get(f, "#3b82f6")}"></span>'
        f"{esc(FAMILY_LABELS.get(f, f.title()))}</div>"
        for f in families
    )
    dots = "".join(
        f'<i style="width:{d}px;height:{d}px"></i>' for d in (5, 8, 11, 14)[: len(sizes)]
    )
    labels = "".join(f"<span>{esc(s)}</span>" for s in sizes)
    return f"""<div class="rg-clg">
  <div class="g"><div class="r"><span class="ln"></span>Pareto frontier</div></div>
  <div class="g"><div class="gt">Chunker</div>{fam_rows}</div>
  <div class="g"><div class="gt">Reranker</div>
    <div class="r"><span class="sw" style="background:{TOKENS['ink_soft']}"></span>ON</div>
    <div class="r"><span class="ring"></span>OFF</div>
  </div>
  <div class="g"><div class="gt">Context size (chars)</div>
    <div class="sizes">{dots}</div><div class="sizelab">{labels}</div>
  </div>
</div>"""


def kpi(
    label: str,
    value: str,
    interval: tuple[float, float] | None = None,
    delta: float | None = None,
    *,
    lower_is_better: bool = False,
    delta_suffix: str = "vs. avg",
) -> str:
    """One metric tile: label, value, confidence interval, delta against the field.

    The interval is shown directly under the value on purpose. Every config
    comparison in this project is noise-limited at the current sample size, so a
    point estimate without its interval is the single easiest way for a reader to
    over-read the table.
    """
    ci = (
        f'<div class="rg-kpi-ci">({interval[0]:.3f} – {interval[1]:.3f})</div>'
        if interval
        else '<div class="rg-kpi-ci">&nbsp;</div>'
    )
    d = '<div class="rg-kpi-d rg-flat">—</div>'
    if delta is not None:
        good = (delta < 0) if lower_is_better else (delta > 0)
        arrow = "↓" if delta < 0 else "↑"
        cls = "rg-up" if good else "rg-down"
        d = f'<div class="rg-kpi-d {cls}">{arrow} {delta:+.1%} <span class="rg-flat">{esc(delta_suffix)}</span></div>'
    return (
        f'<div class="rg-kpi"><div class="rg-kpi-l">{esc(label)}</div>'
        f'<div class="rg-kpi-v">{esc(value)}</div>{ci}{d}</div>'
    )


def kpi_row(tiles: Iterable[str]) -> str:
    return '<div class="rg-kpis">' + "".join(tiles) + "</div>"


def note(title: str, body: str, *, icon: str = "&#128161;", warn: bool = False) -> str:
    cls = "rg-note rg-warn" if warn else "rg-note"
    return (
        f'<div class="{cls}"><div class="rg-note-ico">{icon}</div>'
        f'<div><div class="rg-note-t">{esc(title)}</div>'
        f'<div class="rg-note-b">{body}</div></div></div>'
    )


def prose(text: str) -> str:
    return f'<div class="rg-prose">{text}</div>'


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    winner_row: int | None = 0,
    numeric_from: int = 2,
) -> str:
    """A compact table; `winner_row` gets the highlighted treatment.

    Built as raw HTML rather than st.dataframe because the design needs a
    highlighted leading row, ± intervals rendered smaller inline, and directional
    arrows in the headers -- none of which a dataframe exposes.
    """
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="rg-win"' if winner_row is not None and i == winner_row else ""
        cells = "".join(
            f'<td class="{"rg-num" if j >= numeric_from else ""}">{c}</td>'
            for j, c in enumerate(row)
        )
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<table class="rg-tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def pm(value: float, half_width: float | None, prec: int = 3) -> str:
    """'0.742 ± 0.041' with the interval de-emphasised."""
    if half_width is None:
        return f"{value:.{prec}f}"
    return f'{value:.{prec}f} <span class="rg-pm">± {half_width:.{prec}f}</span>'


def legend(entries: Sequence[tuple[str, str, int, float]]) -> str:
    """(label, colour, count, share) rows beside a donut."""
    rows = []
    for label, colour, count, share in entries:
        rows.append(
            f'<div class="rg-leg-row"><span class="rg-leg-sw" style="background:{colour}"></span>'
            f"<span>{esc(label)}</span>"
            f'<span class="rg-leg-n">{count}</span>'
            f'<span class="rg-leg-p">({share:.0%})</span></div>'
        )
    return '<div class="rg-leg">' + "".join(rows) + "</div>"


def strip(icon: str, title: str, body: str, cta: str | None = None) -> str:
    action = f'<span class="rg-cta">{esc(cta)}</span>' if cta else ""
    return (
        f'<div class="rg-strip"><div class="rg-strip-ico">{icon}</div>'
        f'<div><b>{esc(title)}</b><div class="body">{body}</div></div>{action}</div>'
    )


def sidebar_brand() -> str:
    return (
        '<div class="rg-brand"><div class="rg-brand-mark">◈</div>'
        '<div><div class="rg-brand-name">RAGlign</div>'
        '<div class="rg-brand-sub">RAG Evaluation &amp; Optimization</div></div></div>'
    )


def sidebar_title(text: str) -> str:
    return f'<div class="rg-sb-title">{esc(text)}</div>'


def hint(text: str) -> str:
    return f'<div class="rg-hint"><span>&#9432;</span><span>{text}</span></div>'
