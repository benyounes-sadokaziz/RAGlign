"""Visual theme: design tokens and the CSS that reshapes Streamlit's defaults.

One module owns the palette so the components and the Altair charts cannot drift
apart — the usual failure when chart colours are hardcoded separately from CSS.

Selectors hang off Streamlit's `data-testid` hooks rather than generated class
names, which change between releases.
"""

from __future__ import annotations

TOKENS = {
    # shell
    "shell": "#101b30",
    "sidebar": "#16233a",
    "sidebar_card": "#1d2c47",
    "sidebar_line": "#2b3c59",
    "canvas": "#f6f8fb",
    "card": "#ffffff",
    "card_line": "#e6ebf2",
    # text
    "ink": "#0f172a",
    "ink_soft": "#64748b",
    "ink_faint": "#9aa8bd",
    "on_dark": "#eaf0fb",
    "on_dark_soft": "#8fa0bd",
    # accents
    "blue": "#2f6df6",
    "blue_dim": "#1e4fd8",
    "green": "#22c55e",
    "green_soft": "#e7f9ef",
    "green_line": "#bbf0d0",
    "green_ink": "#15803d",
    "violet": "#8b5cf6",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "pink": "#ec4899",
}

# Chunker families keep one colour across every chart and legend.
FAMILY_COLORS = {
    "semantic": "#22c55e",
    "recursive": "#ec4899",
    "fixed": "#f59e0b",
    "heading": "#8b5cf6",
    "other": "#3b82f6",
}

FAMILY_LABELS = {
    "semantic": "Semantic",
    "recursive": "Recursive",
    "fixed": "Fixed-size",
    "heading": "Structure",
    "other": "Other",
}


def family_of(label: str) -> str:
    head = label.split("/")[0]
    for key in FAMILY_COLORS:
        if head.startswith(key):
            return key
    return "other"


def css() -> str:
    t = TOKENS
    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* ---------- shell: dark ground, light rounded content panel ---------- */
  [data-testid="stAppViewContainer"] {{ background: {t['shell']}; }}
  [data-testid="stHeader"] {{ background: transparent; height: 0; }}
  [data-testid="stMain"] {{
      background: {t['canvas']};
      border-radius: 18px 0 0 0;
      margin-top: 8px;
  }}
  .block-container {{ padding: 1.25rem 1.75rem 2.5rem; max-width: 1640px; }}
  html, body, [class*="css"], button, input, select, textarea {{
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      color: {t['ink']};
  }}
  #MainMenu, footer {{ visibility: hidden; }}

  /* Every text colour in the content panel is stated outright. Inheriting from
     Streamlit's theme meant a viewer in dark mode got white text on the white
     cards this design paints -- an invisible page. */
  [data-testid="stMain"], [data-testid="stMain"] p, [data-testid="stMain"] span,
  [data-testid="stMain"] div, [data-testid="stMain"] td, [data-testid="stMain"] th,
  [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
  [data-testid="stMain"] li, [data-testid="stMain"] label {{ color: {t['ink']}; }}
  [data-testid="stMain"] [data-testid="stMarkdownContainer"] {{ color: {t['ink']}; }}

  /* ---------- sidebar ---------- */
  section[data-testid="stSidebar"] {{ background: {t['sidebar']}; width: 268px !important; }}
  section[data-testid="stSidebar"] > div {{ background: {t['sidebar']}; padding-top: 0; }}
  section[data-testid="stSidebar"] * {{ color: {t['on_dark']}; }}
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .5rem; }}
  section[data-testid="stSidebar"] hr {{
      border: none; border-top: 1px solid {t['sidebar_line']}; margin: .55rem 0 .7rem;
  }}

  .rg-brand {{ display:flex; gap:.72rem; align-items:center; padding: 1.15rem .2rem .95rem; }}
  .rg-brand-mark {{
      width:40px; height:40px; border-radius:12px; flex:none;
      background: linear-gradient(145deg,#7cb0ff 0%, {t['blue']} 55%, #6d4bf0 100%);
      display:flex; align-items:center; justify-content:center; font-size:20px;
      box-shadow: 0 6px 16px rgba(47,109,246,.4);
  }}
  .rg-brand-name {{ font-size:1.3rem; font-weight:800; line-height:1.1; letter-spacing:-.025em; }}
  .rg-brand-sub  {{ font-size:.67rem; color:{t['on_dark_soft']}; font-weight:500; margin-top:1px; }}

  /* nav — radio group restyled as a nav list */
  section[data-testid="stSidebar"] [role="radiogroup"] {{ gap:.18rem; }}
  section[data-testid="stSidebar"] [role="radiogroup"] > label {{
      padding:.62rem .8rem; border-radius:10px; cursor:pointer; width:100%;
      transition: background .13s ease;
  }}
  section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{ background:{t['sidebar_card']}; }}
  section[data-testid="stSidebar"] [role="radiogroup"] > label p {{
      font-size:.875rem; font-weight:500; margin:0; letter-spacing:-.005em;
  }}
  section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
      background:{t['blue']}; box-shadow:0 4px 14px rgba(47,109,246,.45);
  }}
  section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{ font-weight:600; }}
  /* Hide the radio dot by finding the element that CONTAINS the input, rather
     than by position or by exclusion.
     An exclusion rule (`label > *:not([data-testid="stMarkdownContainer"])`)
     looked safe and was not: the markdown container sits one level deeper, so
     the wrapper holding the nav text matched the :not() and the entire menu
     disappeared. `:has(input)` can only ever match the indicator. */
  section[data-testid="stSidebar"] [role="radiogroup"] input,
  section[data-testid="stSidebar"] [role="radiogroup"] label > div:has(> input),
  section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:has(input) {{
      display:none !important;
  }}
  section[data-testid="stSidebar"] [role="radiogroup"] label {{
      display:flex !important; align-items:center !important; gap:0 !important;
  }}
  /* Belt and braces for builds where :has() does not match the wrapper: shrink
     the indicator itself to nothing without touching any text node. */
  section[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] > div:first-child {{
      width:0 !important; height:0 !important; overflow:hidden !important;
      margin:0 !important; border:0 !important;
  }}
  section[data-testid="stSidebar"] [role="radiogroup"] label > div {{ background:transparent !important; }}
  section[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] {{
      display:block !important;
  }}

  /* section header with chevron */
  .rg-sec {{
      display:flex; align-items:center; justify-content:space-between;
      font-size:1.02rem; font-weight:700; padding:.15rem .2rem .1rem; letter-spacing:-.01em;
  }}
  .rg-sec span.chev {{ color:{t['on_dark_soft']}; font-size:.85rem; }}
  .rg-sec-sm {{ font-size:.95rem; margin-top:.85rem; }}
  .rg-lab {{
      font-size:.78rem; font-weight:600; color:{t['on_dark_soft']};
      margin:.55rem .2rem .3rem;
  }}

  /* selector shown as a card; the real select sits invisibly on top */
  .rg-pick {{
      background:{t['sidebar_card']}; border:1px solid {t['sidebar_line']}; border-radius:11px;
      padding:.62rem .7rem; display:flex; gap:.6rem; align-items:center;
  }}
  .rg-pick-ico {{
      width:26px; height:26px; border-radius:7px; background:#26395c; flex:none;
      display:flex; align-items:center; justify-content:center; font-size:13px;
  }}
  .rg-pick-t {{ font-size:.85rem; font-weight:600; line-height:1.2; }}
  .rg-pick-d {{ font-size:.68rem; color:{t['on_dark_soft']}; margin-top:2px; line-height:1.3; }}
  .rg-pick-ch {{ margin-left:auto; color:{t['on_dark_soft']}; font-size:.7rem; }}

  /* Selects. The theme's secondaryBackgroundColor paints these white, while the
     sidebar keeps its light text -- producing an unreadable white-on-white
     control. Both surface and text are therefore forced here, on the wrapper and
     every inner node, since the value sits several divs deep. */
  /* Selects render on a light surface that repeated attempts failed to repaint
     dark -- each one named a different baseweb node and missed whichever
     actually carries it. So the contrast is fixed from the other side: the text
     is forced DARK to match the light control. Readability beats matching the
     surrounding dark panel, and the outcome is a clean white field with a light
     border, which reads deliberately rather than broken. */
  section[data-testid="stSidebar"] div[data-testid="stSelectbox"] *,
  section[data-testid="stSidebar"] div[data-testid="stSelectbox"] input,
  section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
      color:{t['ink']} !important;
      -webkit-text-fill-color:{t['ink']} !important;
      opacity:1 !important;
      font-size:.85rem; font-weight:500;
  }}
  section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
      background-color:#ffffff !important;
      border:1px solid #cfd8e8 !important;
      border-radius:11px !important;
      min-height:42px;
  }}
  section[data-testid="stSidebar"] div[data-testid="stSelectbox"] svg {{
      fill:{t['ink_soft']} !important; color:{t['ink_soft']} !important;
  }}

  /* The dropdown is a light surface too, so its items follow the same rule. */
  [data-baseweb="popover"] li,
  [data-baseweb="popover"] li * {{
      color:{t['ink']} !important; -webkit-text-fill-color:{t['ink']} !important;
  }}
  [data-baseweb="popover"] li:hover,
  [data-baseweb="popover"] li:hover * {{
      background:{t['blue']} !important; color:#fff !important;
      -webkit-text-fill-color:#fff !important;
  }}

  [data-baseweb="popover"] [role="listbox"],
  [data-baseweb="popover"] ul {{
      background:#ffffff !important; border:1px solid #cfd8e8 !important;
  }}

  section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {{ margin-bottom:.2rem; }}
  section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {{
      font-size:.78rem; font-weight:600; color:{t['on_dark_soft']};
  }}

  /* Segmented control -> the tau buttons, in a single row.
     flex-direction and nowrap are stated explicitly: the group inherited a
     column direction from its Streamlit wrapper and the four buttons stacked
     vertically down the sidebar. */
  section[data-testid="stSidebar"] [data-testid="stButtonGroup"] {{
      display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important;
      gap:.35rem; width:100%;
  }}
  section[data-testid="stSidebar"] [data-testid="stButtonGroup"] > div,
  section[data-testid="stSidebar"] [data-testid="stButtonGroup"] > span {{
      flex:1 1 0 !important; min-width:0 !important; display:flex !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {{
      background:{t['sidebar_card']} !important; border:1px solid {t['sidebar_line']} !important;
      border-radius:10px !important; color:{t['on_dark']} !important;
      font-size:.85rem !important; font-weight:600 !important;
      padding:.5rem .2rem !important; width:100% !important; min-width:0 !important;
      height:auto !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"],
  section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {{
      background:{t['blue']} !important; border-color:{t['blue']} !important; color:#fff !important;
      box-shadow:0 4px 12px rgba(47,109,246,.4);
  }}

  /* sliders: thin track, per-slider accent, value chip on the right */
  section[data-testid="stSidebar"] [data-testid="stSlider"] {{ padding:.1rem 0 .15rem; }}
  section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
  section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
  section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {{ display:none; }}
  section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{
      background:#fff !important; box-shadow:0 1px 4px rgba(0,0,0,.4) !important;
      height:14px !important; width:14px !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {{
      height:5px !important;
  }}
  .rg-wrow {{ display:flex; align-items:center; justify-content:space-between; margin:.35rem .2rem -.55rem; }}
  .rg-wrow .n {{ font-size:.8rem; font-weight:500; color:{t['on_dark']}; }}
  .rg-chip {{
      background:{t['sidebar_card']}; border:1px solid {t['sidebar_line']}; border-radius:7px;
      padding:.1rem .45rem; font-size:.74rem; font-weight:600; font-variant-numeric:tabular-nums;
      color:{t['on_dark']};
  }}
  section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child > div:first-child,
  section[data-testid="stSidebar"] [data-testid="stSliderTrack"] > div:first-child {{
      background: {t['blue']} !important;
  }}

  /* Sidebar button (weights reset): quiet by default, blue on hover, and
     visibly inert once the weights are already at their defaults. */
  section[data-testid="stSidebar"] [data-testid="stButton"] button {{
      background:{t['sidebar_card']} !important; border:1px solid {t['sidebar_line']} !important;
      color:{t['on_dark']} !important; border-radius:10px !important;
      font-size:.8rem !important; font-weight:600 !important; padding:.5rem .6rem !important;
      margin-top:.7rem; transition:background .13s ease, border-color .13s ease;
  }}
  section[data-testid="stSidebar"] [data-testid="stButton"] button:hover:not(:disabled) {{
      background:{t['blue']} !important; border-color:{t['blue']} !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stButton"] button:disabled {{
      opacity:.4 !important; cursor:default !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stButton"] button * {{
      color:{t['on_dark']} !important;
  }}

  .rg-hint {{
      background:{t['sidebar_card']}; border:1px solid {t['sidebar_line']}; border-radius:11px;
      padding:.7rem .8rem; font-size:.72rem; line-height:1.5; color:{t['on_dark_soft']};
      display:flex; gap:.5rem; margin-top:.9rem;
  }}

  /* ---------- topbar ---------- */
  .rg-topbar {{ display:flex; justify-content:flex-end; align-items:center; gap:.7rem; margin:-.35rem 0 .5rem; }}
  .rg-pill {{
      display:inline-flex; align-items:center; gap:.45rem; background:{t['card']};
      border:1px solid {t['card_line']}; border-radius:999px; padding:.36rem .85rem;
      font-size:.76rem; color:{t['ink_soft']}; font-weight:500;
  }}
  .rg-dot {{ width:8px; height:8px; border-radius:50%; background:{t['green']}; }}
  .rg-moon {{ color:{t['ink_faint']}; font-size:1.05rem; }}

  /* ---------- page heading ---------- */
  .rg-head {{ display:flex; gap:.85rem; align-items:flex-start; margin:.15rem 0 1.15rem; }}
  .rg-head-icon {{ font-size:1.75rem; line-height:1; margin-top:.1rem; color:{t['blue']}; }}
  .rg-head h1 {{ font-size:1.95rem; font-weight:800; margin:0; letter-spacing:-.03em; }}
  .rg-head p  {{ color:{t['ink_soft']}; margin:.2rem 0 0; font-size:.93rem; }}

  /* ---------- cards ---------- */
  .rg-card {{
      background:{t['card']}; border:1px solid {t['card_line']}; border-radius:15px;
      padding:1.15rem 1.25rem; box-shadow:0 1px 2px rgba(15,23,42,.04); height:100%;
  }}
  .rg-card h3 {{ font-size:1.06rem; font-weight:700; margin:0 0 .1rem; letter-spacing:-.015em; }}
  .rg-card h3 .rg-mute {{ font-weight:500; color:{t['ink_faint']}; font-size:.85rem; }}
  .rg-card .rg-sub {{ color:{t['ink_soft']}; font-size:.83rem; margin:.15rem 0 .9rem; }}

  .rg-badge {{
      display:inline-flex; align-items:center; gap:.45rem; background:{t['green_soft']};
      border:1px solid {t['green_line']}; color:{t['green_ink']}; font-weight:700; font-size:.8rem;
      padding:.36rem .8rem; border-radius:9px; margin-bottom:1.05rem;
  }}
  .rg-spec {{ font-size:.94rem; margin-bottom:1.1rem; }}
  .rg-spec .rg-k {{ color:{t['ink_soft']}; font-weight:500; }}
  .rg-spec b {{ font-weight:700; }}
  .rg-spec .rg-sep {{ color:{t['ink_faint']}; margin:0 .7rem; font-weight:700; }}

  .rg-kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; }}
  .rg-kpi {{ background:#fff; border:1px solid {t['card_line']}; border-radius:12px; padding:.8rem .85rem; }}
  .rg-kpi-l {{ font-size:.75rem; color:{t['ink_soft']}; font-weight:600; margin-bottom:.45rem; }}
  .rg-kpi-v {{ font-size:2rem; font-weight:800; letter-spacing:-.04em; line-height:1; }}
  .rg-kpi-ci {{ font-size:.73rem; color:{t['ink_faint']}; margin-top:.35rem; }}
  .rg-kpi-d  {{ font-size:.74rem; margin-top:.6rem; font-weight:600; }}
  .rg-kpi-d .rg-vs {{ color:{t['ink_faint']}; font-weight:500; margin-left:.2rem; }}
  .rg-up {{ color:{t['green']}; }} .rg-down {{ color:{t['blue']}; }} .rg-flat {{ color:{t['ink_faint']}; }}

  .rg-note {{
      background:{t['green_soft']}; border:1px solid {t['green_line']}; border-radius:12px;
      padding:.85rem .95rem; margin-top:1rem; display:flex; gap:.6rem;
  }}
  .rg-note-ico {{ font-size:1rem; line-height:1.3; }}
  .rg-note-t {{ font-weight:700; font-size:.85rem; margin-bottom:.3rem; color:{t['green_ink']}; }}
  .rg-note-b {{ font-size:.83rem; color:#166534; line-height:1.55; }}
  .rg-warn {{ background:#fffbeb; border-color:#fde68a; }}
  .rg-warn .rg-note-t {{ color:#92400e; }} .rg-warn .rg-note-b {{ color:#a16207; }}
  .rg-prose {{ font-size:.89rem; line-height:1.68; color:#334155; }}

  /* ---------- tables ---------- */
  .rg-tbl {{ width:100%; border-collapse:separate; border-spacing:0; font-size:.855rem; }}
  .rg-tbl th {{
      text-align:left; font-weight:600; color:{t['ink_soft']}; font-size:.79rem;
      padding:.6rem .7rem; border-bottom:1px solid {t['card_line']}; white-space:nowrap;
  }}
  .rg-tbl td {{ padding:.62rem .7rem; border-bottom:1px solid #f2f5f9; white-space:nowrap; }}
  .rg-tbl tr:last-child td {{ border-bottom:none; }}
  .rg-tbl .rg-num {{ font-variant-numeric:tabular-nums; }}
  .rg-tbl .rg-pm {{ color:{t['ink_faint']}; font-size:.79rem; }}
  .rg-tbl tr.rg-win td {{ background:{t['green_soft']}; }}
  .rg-tbl tr.rg-win td:first-child {{ border-left:3px solid {t['green']}; border-radius:8px 0 0 8px; }}
  .rg-tbl tr.rg-win td:last-child {{ border-radius:0 8px 8px 0; }}
  .rg-tbl tr:hover td {{ background:#f8fafc; }}
  .rg-tbl tr.rg-win:hover td {{ background:#dcf6e7; }}
  .rg-rank {{ color:{t['ink_faint']}; font-variant-numeric:tabular-nums; }}

  /* ---------- chart legend (right of the scatter) ---------- */
  .rg-clg {{ font-size:.73rem; }}
  .rg-clg .g {{ margin-bottom:.7rem; }}
  .rg-clg .gt {{ color:{t['ink_soft']}; font-weight:600; margin-bottom:.35rem; }}
  .rg-clg .r {{ display:flex; align-items:center; gap:.45rem; margin-bottom:.22rem; color:{t['ink']}; }}
  .rg-clg .sw {{ width:9px; height:9px; border-radius:50%; flex:none; }}
  .rg-clg .ln {{ width:16px; height:2px; background:{t['blue']}; flex:none; }}
  .rg-clg .ring {{ width:9px; height:9px; border-radius:50%; border:1.5px solid {t['ink_soft']}; flex:none; }}
  .rg-clg .sizes {{ display:flex; align-items:center; gap:.3rem; margin-top:.15rem; }}
  .rg-clg .sizes i {{ border-radius:50%; background:{t['ink_faint']}; display:block; }}
  .rg-clg .sizelab {{ display:flex; gap:.55rem; color:{t['ink_faint']}; margin-top:.2rem; font-size:.68rem; }}

  /* ---------- donut legend ---------- */
  .rg-leg {{ display:flex; flex-direction:column; gap:.55rem; padding-top:.4rem; }}
  .rg-leg-row {{ display:flex; align-items:center; gap:.55rem; font-size:.84rem; }}
  .rg-leg-sw {{ width:11px; height:11px; border-radius:3px; flex:none; }}
  .rg-leg-n {{ margin-left:auto; font-weight:700; font-variant-numeric:tabular-nums; }}
  .rg-leg-p {{ color:{t['ink_faint']}; width:44px; text-align:right; font-variant-numeric:tabular-nums; }}

  /* ---------- key insight strip ---------- */
  .rg-strip {{
      background:{t['card']}; border:1px solid {t['card_line']}; border-radius:15px;
      padding:1rem 1.2rem; display:flex; gap:.85rem; align-items:center;
      box-shadow:0 1px 2px rgba(15,23,42,.04);
  }}
  .rg-strip-ico {{ font-size:1.2rem; color:{t['blue']}; }}
  .rg-strip b {{ font-weight:700; font-size:.92rem; }}
  .rg-strip .body {{ font-size:.86rem; color:{t['ink_soft']}; margin-top:.15rem; }}
  .rg-hl {{ color:{t['blue']}; font-weight:600; }}
  .rg-cta {{
      margin-left:auto; background:{t['blue']}; color:#fff !important; border-radius:10px;
      padding:.6rem 1.1rem; font-size:.86rem; font-weight:600; white-space:nowrap;
      box-shadow:0 4px 14px rgba(47,109,246,.35);
  }}

  .rg-gap {{ height:.9rem; }}
  div[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid {t['card_line']}; }}
  [data-testid="stExpander"] details {{
      border:1px solid {t['card_line']}; border-radius:12px; background:{t['card']};
  }}
</style>
"""
