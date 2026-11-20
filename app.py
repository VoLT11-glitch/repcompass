"""Repcompass: turn a Lyfta export into clear answers about your training.

Run:  python -m streamlit run app.py
"""
import json
import os
from datetime import date, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as A
import features as F
import importers as I
import report as R
import browser_store as BS
import lyfta_api as L

st.set_page_config(page_title="Repcompass", page_icon="🏋️", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ look & feel
INK, INK2, MUTED = "#f2f1ec", "#c4c3b8", "#8b8a82"
SURFACE, CARD, GRID = "#171716", "#1b1b1a", "#2a2a27"
BLUE, ORANGE, AQUA = A.SERIES[0], A.SERIES[1], A.SERIES[2]
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"

st.markdown(
    """
<style>
:root { --bg:#0f0f0e; --s1:#171716; --s2:#1d1d1b; --line:#2a2a27; --line2:#353531;
        --ink:#f2f1ec; --ink2:#c4c3b8; --muted:#8b8a82; --acc:#4b93ec; --acc-soft:rgba(57,135,229,.14);
        --good:#4cc76a; --warn:#f2b33d; --crit:#ff7b7b; --r:16px; --ease:cubic-bezier(.2,.7,.2,1); }
.stApp { background: radial-gradient(900px 420px at 0% -8%, rgba(57,135,229,.16) 0%, rgba(15,15,14,0) 62%), var(--bg); }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { letter-spacing: -0.02em; }
::selection { background: rgba(75,147,236,.35); }

/* ---- top bar */
.brand { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.05rem; letter-spacing:-.01em; }
.brand .mark { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
               background: linear-gradient(140deg, #4b93ec, #2a5fb0); box-shadow: 0 6px 18px rgba(57,135,229,.35); font-size:.95rem; }
.brand .tag { color:var(--muted); font-weight:500; font-size:.82rem; margin-left:2px; }
.dash-title { margin: 18px 0 2px 0; }
.dash-title h1 { font-size: 2.1rem; font-weight: 800; margin: 0; line-height: 1.1; }
.dash-title p { color: var(--ink2); margin: 6px 0 0 0; font-size: .95rem; }
.dash-title .pill { display:inline-flex; align-items:center; gap:6px; color:var(--ink2); background:var(--s1);
                    border:1px solid var(--line); border-radius:999px; padding:3px 10px; font-size:.78rem; margin-right:6px; }
.dash-title .pill.demo { color:#f7c768; border-color:#5b4614; background:#231b08; }

/* ---- welcome */
.hero { padding: 26px 0 8px 0; }
.hero .kicker { color: #7fb2f0; font-weight: 700; font-size: .74rem; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { font-size: 3rem; font-weight: 850; margin: .3rem 0 .5rem 0; line-height: 1.02; letter-spacing:-.035em; }
.hero h1 .hl { background: linear-gradient(90deg, #8cc0ff, #4b93ec 60%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: var(--ink2); font-size: 1.05rem; max-width: 640px; margin: 0; line-height:1.55; }
.panel-h { font-weight: 750; font-size: 1.02rem; margin: 2px 0 2px 0; }
.panel-s { color: var(--muted); font-size: .85rem; margin-bottom: 10px; }
.steps { counter-reset: s; list-style: none; padding: 0; margin: 4px 0 0 0; }
.steps li { position: relative; padding: 0 0 14px 40px; color: var(--ink2); font-size: .9rem; line-height: 1.45; }
.steps li b { color: var(--ink); }
.steps li::before { counter-increment: s; content: counter(s); position:absolute; left:0; top:-2px; width:26px; height:26px;
                    border-radius:8px; display:grid; place-items:center; font-weight:800; font-size:.8rem;
                    background: var(--acc-soft); color:#8cc0ff; border:1px solid #234a7a; }
.feats { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }
.feat { background: var(--s1); border:1px solid var(--line); border-radius: 14px; padding: 14px 16px; }
.feat .i { font-size: 1.2rem; }
.feat b { display:block; margin-top: 6px; font-size: .92rem; }
.feat span { color: var(--muted); font-size: .82rem; line-height:1.4; }
.trust { color: var(--muted); font-size: .8rem; margin: 4px 0 6px 0; display:flex; gap:16px; flex-wrap:wrap; }

/* ---- containers drawn by Streamlit (st.container(border=True)) */
.st-key-welcome_load, .st-key-welcome_how { background: var(--s1); border-radius: var(--r) !important; }

/* ---- headline numbers */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 6px; }
.kpi { position: relative; background: linear-gradient(180deg, var(--s2), var(--s1)); border: 1px solid var(--line); border-radius: var(--r);
       padding: 15px 16px 12px 16px; overflow: hidden; transition: border-color .2s var(--ease), transform .2s var(--ease); }
.kpi:hover { border-color: var(--line2); transform: translateY(-1px); }
.kpi .l { color: var(--muted); font-size: .72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; }
.kpi .vr { display:flex; align-items:baseline; gap:8px; margin-top: 6px; flex-wrap: wrap; }
.kpi .v { font-size: 1.9rem; font-weight: 800; line-height: 1.05; letter-spacing:-.02em; }
.kpi .s { color: #a9a89f; font-size: .8rem; margin-top: 4px; }
.kpi svg.sp { display:block; width: 100%; height: 30px; margin-top: 8px; }
.dl { display:inline-flex; align-items:center; gap:3px; font-size:.74rem; font-weight:700; padding:1px 7px; border-radius:999px; }
.dl.up { color: var(--good); background: rgba(76,199,106,.12); }
.dl.down { color: var(--crit); background: rgba(255,123,123,.12); }
.dl.flat { color: var(--muted); background: rgba(139,138,130,.14); }

/* ---- insight strip */
.insight { background: linear-gradient(90deg, rgba(57,135,229,.12), rgba(57,135,229,.02) 70%); border: 1px solid #22406a;
           border-radius: var(--r); padding: 12px 16px; margin: 14px 0 4px 0; }
.insight .h { color:#8cc0ff; font-size:.72rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin-bottom:4px; }
.insight .it { display:flex; gap:10px; align-items:flex-start; color: var(--ink); font-size:.92rem; padding: 5px 0; line-height:1.4; }
.insight .it .dot { flex: 0 0 8px; height: 8px; border-radius: 50%; margin-top: 6px; }

/* ---- sections */
.sec { margin: 34px 0 6px 0; }
.sec h3 { font-size: 1.22rem; font-weight: 750; margin: 0; padding: 0; }
.sec .what { color: var(--ink2); font-size: .92rem; margin-top: 4px; max-width: 820px; line-height:1.5; }

/* "How to read this": one tidy fold everywhere instead of three boxes */
details.guide { margin: 8px 0 12px 0; border: 1px solid var(--line); border-radius: 12px; background: var(--s1);
                transition: border-color .2s var(--ease); }
details.guide:hover { border-color: var(--line2); }
details.guide summary { list-style: none; cursor: pointer; padding: 8px 14px; color: #8cc0ff; font-weight: 600; font-size: .82rem;
                        display:flex; align-items:center; gap:8px; user-select:none; }
details.guide summary::-webkit-details-marker { display: none; }
details.guide summary::before { content: ""; width: 7px; height: 7px; border-right: 2px solid currentColor; border-bottom: 2px solid currentColor;
                                transform: rotate(-45deg); transition: transform .2s var(--ease); margin-right: 2px; }
details.guide[open] summary::before { transform: rotate(45deg); }
details.guide .g { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; border-top: 1px solid var(--line);
                   animation: rc-in .25s var(--ease); }
details.guide .g > div { padding: 10px 14px; font-size: .86rem; color: var(--ink2); line-height:1.45; }
details.guide .g > div + div { border-left: 1px solid var(--line); }
details.guide .g b { display: block; color: var(--muted); font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 3px; }

/* ---- charts: one card per chart */
div[data-testid="stPlotlyChart"] { background: var(--s1); border: 1px solid var(--line); border-radius: var(--r); padding: 4px 2px 0 2px;
                                   overflow: hidden; margin-top: 4px; }
.js-plotly-plot .nsewdrag, .js-plotly-plot .draglayer * { cursor: default !important; }

/* ---- chips & cards */
.chip { display: inline-flex; align-items:center; gap:4px; padding: 2px 9px; border-radius: 999px; font-size: .74rem; font-weight: 650; border: 1px solid; white-space: nowrap; }
.chip.good { color: #6fdc6f; border-color: #1c5f1c; background: #0f2a0f; }
.chip.warn { color: #fab219; border-color: #6b4d09; background: #2a2008; }
.chip.crit { color: #ff8a8a; border-color: #6b2a2a; background: #2a1010; }
.chip.info { color: #8cc0ff; border-color: #1f3a5f; background: #0f1b2d; }

.card { background: var(--s1); border: 1px solid var(--line); border-radius: var(--r); padding: 16px 18px; margin-bottom: 12px;
        transition: border-color .2s var(--ease), transform .2s var(--ease); }
.card:hover { border-color: var(--line2); }
.card h4 { margin: 0 0 6px 0; font-size: 1rem; font-weight: 700; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.card .row { color: #a9a89f; font-size: .84rem; margin-top: 8px; line-height:1.5; }
.card .big { font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; }
.card .why { color: var(--ink2); font-size: .84rem; margin-top: 10px; border-top: 1px solid var(--line); padding-top: 9px; line-height:1.45; }
.lines { line-height: 2.2; }

.bar { height: 8px; background: #2a2a27; border-radius: 6px; overflow: hidden; margin: 10px 0 4px 0; }
.bar > div { height: 100%; background: linear-gradient(90deg, #2f6fc4, #5aa2f5); border-radius: 6px; transform-origin: left;
             animation: rc-grow .7s var(--ease); }
.plan { width: 100%; border-collapse: collapse; font-size: .92rem; margin-top: 6px; }
.plan th { text-align: left; color: var(--muted); font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; padding: 6px 8px; border-bottom: 1px solid var(--line); }
.plan td { padding: 11px 8px; border-bottom: 1px solid #222220; color: #e6e5df; vertical-align: top; }
.plan tr:last-child td { border-bottom: 0; }
.plan td.t { font-weight: 800; white-space: nowrap; color: #fff; }
.plan td.n { color: #a9a89f; font-size: .82rem; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ---- tabs: icon + label, clear active state */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; }
[data-testid="stTab"] { border-radius: 10px 10px 0 0; padding: 8px 12px !important; transition: background .15s var(--ease), color .15s var(--ease); }
[data-testid="stTab"]:hover { background: rgba(255,255,255,.04); }
[data-testid="stTab"] p { font-weight: 600; font-size: .92rem; }
[data-testid="stTab"] p > span[role="img"] { font-size: 1.1rem; margin-right: 2px; }
[data-testid="stTab"][aria-selected="true"] p { color: #8cc0ff; }
[data-testid="stTab"] .react-aria-SelectionIndicator { background: #4b93ec; }
[data-testid="stTabPanel"] { animation: rc-in .28s var(--ease); }
.st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: center; }
.st-key-topbar [data-testid="stColumn"] { min-width: 0 !important; }

/* ---- buttons & inputs: visible feedback */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button { transition: transform .12s var(--ease), border-color .15s, background .15s; }
.stButton button:active, .stDownloadButton button:active, [data-testid="stPopover"] button:active { transform: scale(.97); }
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed #3a5f8f !important; background: rgba(57,135,229,.06) !important;
                                         border-radius: 14px !important; transition: background .2s, border-color .2s; }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(57,135,229,.12) !important; border-color: #4b93ec !important; }

.st-key-rc_store { position: absolute !important; width: 0 !important; height: 0 !important; overflow: hidden; }

@keyframes rc-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes rc-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

@media (max-width: 900px) { .feats { grid-template-columns: repeat(2, 1fr); } }

/* ---------------- phones ---------------- */
@media (max-width: 640px) {
  .block-container { padding: 4.2rem 1rem 3rem 1rem; }
  .brand .tag { display: none; }
  .st-key-topbar [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; width: auto !important; }
  .st-key-topbar [data-testid="stColumn"]:last-child { flex: 0 0 auto !important; width: auto !important; }
  header[data-testid="stHeader"] { background: rgba(15,15,14,.94); backdrop-filter: blur(8px); border-bottom: 1px solid #23231f; }
  .hero { padding-top: 10px; }
  .hero h1 { font-size: 2.15rem; }
  .hero p { font-size: .97rem; }
  .feats { gap: 8px; }
  .feat { padding: 12px; }
  .dash-title h1 { font-size: 1.6rem; }

  .kpis { grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .kpi { padding: 12px 13px 10px 13px; border-radius: 14px; }
  .kpi .v { font-size: 1.5rem; }
  .kpi .l { font-size: .64rem; }
  .kpi .s { font-size: .74rem; }
  .kpi svg.sp { height: 24px; }

  details.guide .g { grid-template-columns: 1fr; }
  details.guide .g > div + div { border-left: 0; border-top: 1px solid var(--line); }
  .sec { margin-top: 28px; }
  .sec h3 { font-size: 1.12rem; }
  .sec .what { font-size: .88rem; }

  /* tabs become a 4 x 2 grid of icon tiles, all visible at once */
  [data-testid="stTabs"] [role="tablist"] { display: grid !important; grid-template-columns: repeat(4, 1fr); gap: 6px;
                                            overflow: visible !important; border: 0 !important; box-shadow: none !important; padding-bottom: 6px; }
  [data-testid="stTab"] { margin: 0 !important; padding: 8px 2px !important; height: auto; min-height: 58px; border-radius: 12px;
                          border: 1px solid var(--line) !important; background: var(--s1); box-shadow: none !important; justify-content: center; }
  [data-testid="stTab"] > div, [data-testid="stTab"] [data-testid="stMarkdownContainer"] { width: 100%; }
  [data-testid="stTab"] p { display: flex; flex-direction: column; align-items: center; gap: 3px; font-size: .7rem !important;
                            line-height: 1.15; text-align: center; white-space: normal; }
  [data-testid="stTab"] p > span[role="img"] { font-size: 1.35rem !important; margin: 0 !important; }
  [data-testid="stTab"] .react-aria-SelectionIndicator { display: none !important; }
  [data-testid="stTab"]::after, [data-testid="stTab"]::before,
  [data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] [role="tablist"]::before { display: none !important; }
  [data-testid="stTab"][aria-selected="true"] { background: var(--acc-soft); border-color: #3e7fd0 !important; }
  [data-testid="stTab"][aria-selected="true"] p { color: #fff !important; }
  [data-testid="stTabsScrollRight"], [data-testid="stTabsScrollLeft"] { display: none !important; }

  /* next-workout table becomes one card per lift */
  .plan, .plan tbody { display: block; }
  .plan tr:first-child { display: none; }
  .plan tr { display: grid; grid-template-columns: auto 1fr; column-gap: 10px; align-items: baseline;
             padding: 12px 0; border-bottom: 1px solid #222220; }
  .plan tr:last-child { border-bottom: 0; }
  .plan td { display: block; padding: 0; border: 0; }
  .plan td.l { order: 1; grid-column: 1 / -1; font-weight: 700; margin-bottom: 4px; }
  .plan td.t { order: 2; font-size: 1.35rem; }
  .plan td.s { order: 3; color: #a9a89f; font-size: .85rem; }
  .plan td.s::after { content: " sets"; }
  .plan td.n { order: 4; grid-column: 1 / -1; margin-top: 4px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def section(title: str, what: str):
    st.markdown(f'<div class="sec"><h3>{title}</h3><div class="what">{what}</div></div>', unsafe_allow_html=True)


def guide(read: str, good: str, act: str):
    """How to read it / what good looks like / what to do, folded behind one tap so the page stays calm."""
    st.markdown(
        f'<details class="guide"><summary>How to read this</summary><div class="g">'
        f'<div><b>How to read it</b>{read}</div><div><b>What good looks like</b>{good}</div>'
        f'<div><b>What to do</b>{act}</div></div></details>',
        unsafe_allow_html=True,
    )


def spark(values, color="#4b93ec") -> str:
    """Tiny inline trend line (SVG) for a headline number."""
    v = [float(x) for x in values]
    if len(v) < 2 or max(v) == min(v) == 0:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = [(i * 100 / (len(v) - 1), 26 - (x - lo) / span * 22) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,30 {line} 100,30"
    gid = f"g{abs(hash((tuple(v), color))) % 10**8}"
    return (f'<svg class="sp" viewBox="0 0 100 30" preserveAspectRatio="none">'
            f'<defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".28"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="{area}" fill="url(#{gid})"/><polyline points="{line}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def delta_badge(d, fmt="{:+.0f}", good_up=True) -> str:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return ""
    kind = "flat" if abs(d) < 1e-9 else (("up" if d > 0 else "down") if good_up else ("down" if d > 0 else "up"))
    arrow = "→" if kind == "flat" else ("↑" if d > 0 else "↓")
    return f'<span class="dl {kind}">{arrow} {fmt.format(d)}</span>'


def kpi_row(items):
    """Headline numbers: 4 across on desktop, 2 x 2 on phones.
    items = [(label, value, sub), ...] or (label, value, sub, delta_html, spark_html)."""
    cards = []
    for it in items:
        label, value, sub = it[:3]
        dl = it[3] if len(it) > 3 else ""
        sp = it[4] if len(it) > 4 else ""
        cards.append(f'<div class="kpi"><div class="l">{label}</div><div class="vr"><div class="v">{value}</div>{dl}</div>'
                     f'<div class="s">{sub}</div>{sp}</div>')
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def chip(kind, text):
    icons = {"good": "✓", "warn": "!", "crit": "▼", "info": "•"}
    return f'<span class="chip {kind}">{icons[kind]} {text}</span>'


def style(fig, height=380, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=12, r=16, t=40 if legend else 24, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK2, size=12),
        hoverlabel=dict(bgcolor="#252523", bordercolor="#3a3a37", font=dict(color=INK, size=12)),
        # tapping a legend item shows/hides that series; double-tap isolation is off because it confuses on phones
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK2, size=11),
                    bgcolor="rgba(0,0,0,0)", traceorder="normal", itemclick="toggle", itemdoubleclick=False),
        showlegend=legend, barcornerradius=4, hovermode="closest", hoverdistance=40,
        # No drag-to-zoom: on touch screens a swipe or a slightly moved tap zoomed into an empty box, which
        # looked like the chart vanished until a double-tap reset it. Charts are read, not explored.
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED), zeroline=False, automargin=True, fixedrange=True)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, tickfont=dict(color=MUTED), linecolor="rgba(0,0,0,0)",
                     automargin=True, fixedrange=True)
    return fig


PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "showTips": False, "responsive": True}


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG)


# ------------------------------------------------------------------ data
here = Path(__file__).parent
auto = sorted([*here.glob("export*.csv"), *(here.parent / "data").glob("export*.csv")],
              key=lambda p: p.stat().st_mtime, reverse=True)
if os.environ.get("REPCOMPASS_NO_LOCAL"):  # tests: behave like the public site, which has no local file
    auto = []
# Invisible browser-storage component, mounted at the end of the run. It lives in the main area because
# on phones a closed sidebar is not rendered at all, so anything in it would never run.
store_slot = st.container(key="rc_store")

# Settings widgets live in the Settings tab, which is not drawn on the welcome screen. Streamlit drops a
# widget's value when the widget is not drawn, so re-assign them every run to keep them.
KEEP = ("remember", "unit", "low", "high", "step", "bw", "sex", "lyfta_key", "remember_key", "demo")
for _k in KEEP:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]


def apply_setup(s: dict):
    """Restore goals, muscle fixes and settings. Must run before the widgets are created."""
    if not isinstance(s, dict):
        return
    if isinstance(s.get("goals"), list):
        st.session_state["goals"] = [g for g in s["goals"] if isinstance(g, dict) and {"exercise", "target"} <= set(g)]
    if isinstance(s.get("muscle_overrides"), dict):
        st.session_state["muscle_overrides"] = {k: v for k, v in s["muscle_overrides"].items() if v in F.MUSCLE_CHOICES}
    if isinstance(s.get("bodyweight"), (int, float)):
        st.session_state["bw"] = float(s["bodyweight"])
    if s.get("sex") in ("Male", "Female"):
        st.session_state["sex"] = s["sex"]
    if s.get("unit") in ("kg", "lb"):
        st.session_state["unit"] = s["unit"]
    for k, lo, hi in (("low", 3, 20), ("high", 4, 30)):
        if isinstance(s.get(k), int) and lo <= s[k] <= hi:
            st.session_state[k] = s[k]
    if isinstance(s.get("step"), (int, float)) and 0.5 <= s["step"] <= 20:
        st.session_state["step"] = float(s["step"])


def current_setup() -> dict:
    return {"app": "repcompass", "version": 2, "goals": st.session_state.get("goals", []),
            "muscle_overrides": st.session_state.get("muscle_overrides", {}),
            "bodyweight": st.session_state.get("bw"), "sex": st.session_state.get("sex"),
            "unit": st.session_state.get("unit"), "low": st.session_state.get("low"),
            "high": st.session_state.get("high"), "step": st.session_state.get("step")}


def forget_device():
    BS.clear_all()
    BS.put("optout", "1")
    st.session_state["remember"] = False
    st.session_state["remember_key"] = False
    st.session_state["lyfta_key"] = ""
    st.session_state.pop("lyfta_sync", None)


def run_sync(key: str):
    try:
        with st.spinner("Syncing your Lyfta workouts..."):
            ws = L.fetch_workouts(key)
            f, info = L.to_frame(ws, getattr(st.context, "timezone", None))
        st.session_state["lyfta_sync"] = (f, info)
        st.session_state["lyfta_sync_at"] = pd.Timestamp.now().strftime("%d %b %H:%M")
        st.session_state.pop("lyfta_err", None)
    except L.LyftaError as exc:
        st.session_state["lyfta_err"] = str(exc)


def start_demo():
    st.session_state["demo"] = True


def stop_demo():
    st.session_state["demo"] = False


# restore what this browser saved last time (once per session, before any widget exists)
saved = BS.items()
if saved is not None and not st.session_state.get("_restored"):
    st.session_state["_restored"] = True
    try:
        apply_setup(json.loads(saved.get("setup") or "{}"))
    except (ValueError, TypeError):
        pass
    st.session_state["remember"] = saved.get("optout") != "1"
    if saved.get("lyfta_key"):
        st.session_state["lyfta_key"] = saved["lyfta_key"]
        st.session_state["remember_key"] = True
st.session_state.setdefault("remember", True)
st.session_state.setdefault("demo", False)
st.session_state.setdefault("lyfta_key", "")
st.session_state.setdefault("remember_key", False)
st.session_state.setdefault("unit", "kg")
st.session_state.setdefault("low", 8)
st.session_state.setdefault("high", 12)
st.session_state.setdefault("step", 2.5 if st.session_state["unit"] == "kg" else 5.0)


def _unit_changed():
    st.session_state["step"] = 2.5 if st.session_state["unit"] == "kg" else 5.0


use_demo = st.session_state["demo"]
remember = st.session_state["remember"]
unit = st.session_state["unit"]
low, high, step = st.session_state["low"], st.session_state["high"], st.session_state["step"]
lyfta_key = st.session_state.get("lyfta_key") or ""

UPLOAD_HELP = ("Lyfta, Hevy or Strong CSV. Add several files to combine them. "
               "A workout that appears in more than one file is only counted once.")


def uploader():
    # Same key and arguments wherever it is drawn (welcome card or the "Add workouts" menu), so Streamlit
    # treats it as one widget and the chosen files survive the switch from welcome screen to dashboard.
    return st.file_uploader("Workout export (CSV)", type="csv", accept_multiple_files=True,
                            label_visibility="collapsed", key="uploads", help=UPLOAD_HELP)


# a remembered key syncs once per visit by itself
if (saved is not None and lyfta_key and st.session_state.get("remember_key")
        and not st.session_state.get("_auto_synced") and not use_demo):
    st.session_state["_auto_synced"] = True
    run_sync(lyfta_key)

# Will there be something to show? Decides between the welcome screen and the dashboard before loading,
# because the uploader has to be drawn (in one place or the other) before its files can be read.
has_saved = bool(remember and saved and saved.get("history"))
expect_data = bool(use_demo or st.session_state.get("uploads") or has_saved or auto or st.session_state.get("lyfta_sync"))

top = st.container()
topbar = top.container(key="topbar")
with topbar:
    bar = st.columns([3, 1.25], vertical_alignment="center")
    bar[0].markdown('<div class="brand"><div class="mark">🏋️</div>Repcompass<span class="tag">· training, explained</span></div>',
                    unsafe_allow_html=True)
ups = None
if expect_data:
    with bar[1].popover("Add workouts", icon=":material/upload_file:", width="stretch"):
        st.markdown('<div class="panel-h">Add newer workouts</div>'
                    '<div class="panel-s">Drop your latest export. It is merged with what you already have, '
                    'and repeated workouts are only counted once.</div>', unsafe_allow_html=True)
        ups = uploader()
        if use_demo:
            st.button("Leave demo and use my data", on_click=stop_demo, width="stretch", icon=":material/logout:")
        elif has_saved:
            st.caption("Your history is remembered in this browser, so you only need the newest export each time.")

raw_export, infos, load_err, saved_rows, df = None, [], None, 0, None
if use_demo:
    df = A.load_demo()
elif expect_data:
    frames, infos = [], []
    hist = I.unpack_history(saved["history"]) if has_saved else None
    if hist is not None and len(hist):
        saved_rows = len(hist)
        frames.append(hist)
        infos.append({"source": "Saved on this device", "unit": None, "rows": len(hist), "name": "Saved history"})
    synced = st.session_state.get("lyfta_sync")
    try:
        if not ups and not synced and auto:
            f, info = I.read_export(auto[0])
            info["name"] = auto[0].name
            frames.append(f)
            infos.append(info)
        if synced:
            frames.append(synced[0])
            infos.append(synced[1])
        for u in ups or []:
            f, info = I.read_export(u)
            info["name"] = u.name
            frames.append(f)
            infos.append(info)
        df, raw_export, infos = I.combine(frames, infos) if frames else (None, None, [])
    except Exception as exc:  # unreadable or unknown file
        df, load_err = None, str(exc)
    # keep the combined history in this browser when anything new came in
    if remember and raw_export is not None and len(frames) > (1 if saved_rows else 0):
        blob = I.pack_history(raw_export)
        if len(blob) < 4_000_000:
            BS.put("history", blob)
if not use_demo:
    if not remember and saved and saved.get("history"):
        BS.put("history", None)
    BS.put("optout", None if remember else "1")
    BS.put("lyfta_key", lyfta_key if (st.session_state.get("remember_key") and lyfta_key) else None)


def flush_store():
    """Send queued browser-storage writes. Runs at the end of every full run."""
    with store_slot:
        err = BS.mount()
    if err:
        st.caption(f"This browser would not save your data ({err}).")


# ------------------------------------------------------------------ welcome (no data yet)
def welcome():
    st.markdown(
        '<div class="hero"><div class="kicker">Free · no account · works with Lyfta, Hevy and Strong</div>'
        "<h1>Know if your training is <span class='hl'>working</span>.</h1>"
        "<p>Drop in your workout export and get straight answers: which lifts are climbing, which have stalled, "
        "whether every muscle gets enough work, and exactly what to lift next session.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1.container(key="welcome_load", border=True):
        st.markdown('<div class="panel-h">Load your workouts</div>'
                    '<div class="panel-s">Choose the CSV file you exported. On a phone, tap Upload and pick it '
                    'from Files or Downloads.</div>', unsafe_allow_html=True)
        got = uploader()
        if load_err:
            st.error(load_err)
        st.button("Try it with demo data", on_click=start_demo, icon=":material/play_arrow:", width="stretch")
        st.markdown('<div class="trust"><span>🔒 Read in memory, never stored on a server</span>'
                    '<span>💾 Remembered in this browser only</span></div>', unsafe_allow_html=True)
    with c2.container(key="welcome_how", border=True):
        st.markdown(
            '<div class="panel-h">Get your export from Lyfta</div>'
            '<div class="panel-s">About 30 seconds, and only needed once. No API key, no login here.</div>'
            '<ol class="steps">'
            '<li><b>Open Lyfta → Settings → Export data.</b> It saves a CSV with every workout you have logged.</li>'
            '<li><b>Load that file here.</b> Repcompass reads it in your browser session and keeps a copy in '
            'this browser so next time it opens straight to your dashboard.</li>'
            '<li><b>Later, add only the new export.</b> Repcompass merges it and never counts a workout twice.</li>'
            '</ol>', unsafe_allow_html=True)
        with st.expander("Using Hevy or Strong?"):
            st.markdown("- **Hevy:** Settings → Export & Import Data → Export workouts.\n"
                        "- **Strong:** Settings → Export Strong data.\n\n"
                        "Load the CSV the same way. You can mix files from different apps.")
    st.markdown(
        '<div class="feats">'
        '<div class="feat"><div class="i">📈</div><b>Lift progress</b><span>Estimated max for every lift, with PRs marked.</span></div>'
        '<div class="feat"><div class="i">🧱</div><b>Plateau check</b><span>Spots stalled lifts and says what to change.</span></div>'
        '<div class="feat"><div class="i">🎯</div><b>Next workout</b><span>Weight × reps targets for every lift.</span></div>'
        '<div class="feat"><div class="i">💪</div><b>Muscle balance</b><span>Hard sets per muscle against a 10-20 band.</span></div>'
        '</div>', unsafe_allow_html=True)
    return got


if df is None:
    if saved is None and not use_demo and not st.session_state.get("uploads") and "_store_checked" not in st.session_state:
        # first run of a visit: the browser has not answered yet, so do not flash the welcome screen
        st.session_state["_store_checked"] = True
        st.caption("Checking this browser for saved data...")
    elif expect_data:
        # something was expected but could not be read (for example a broken file)
        if load_err:
            st.error(f"{load_err} Open Add workouts to try another file, or remove the file there.")
        else:
            st.info("No workouts found. Open Add workouts to load an export.")
    else:
        welcome()
    flush_store()
    st.stop()


# ------------------------------------------------------------------ shared numbers
overrides = st.session_state.get("muscle_overrides", {})
df = F.apply_muscle_overrides(df, overrides)
data_key = (len(df), str(df["Date"].min()), str(df["Date"].max()),
            float(df["Weight"].fillna(0).sum()), float(df["Reps"].fillna(0).sum()))
w = A.working_sets(df)
sess = df.drop_duplicates("session")
last_day = sess["day"].max()
first_day = sess["day"].min()
span_weeks = max((last_day - first_day).days / 7.0, 1)
l28 = sess[sess["day"] > last_day - pd.Timedelta(days=28)]
p28 = sess[(sess["day"] <= last_day - pd.Timedelta(days=28)) & (sess["day"] > last_day - pd.Timedelta(days=56))]
lifts = (w[w["e1rm"].notna()].groupby("Exercise")["day"].nunique().sort_values(ascending=False))
plateaus = A.plateau_report(df)
n_stalled = int((plateaus["Status"] == "Stalled").sum()) if len(plateaus) else 0
prs = A.pr_timeline(df)
deload = F.deload_check(df)
bal = A.muscle_balance(df)
review = F.weekly_review(df, plateaus, deload, bal, st.session_state.get("goals", []), unit)

with top:
    pills = (f'<span class="pill">📅 {first_day:%d %b %Y} – {last_day:%d %b %Y}</span>'
             f'<span class="pill">🏋️ {len(sess)} workouts</span>')
    if use_demo:
        pills = '<span class="pill demo">Demo data</span>' + pills
    elif saved_rows:
        pills += '<span class="pill">💾 Remembered on this device</span>'
    st.markdown(f'<div class="dash-title"><h1>Your training, explained.</h1><p>{pills}</p></div>', unsafe_allow_html=True)
    for info in infos:
        if info.get("unit") and info["unit"] != unit:
            st.warning(f"This {info['source']} export is in {info['unit']}. Switch the unit to {info['unit']} in Settings.")

t_over, t_lift, t_plat, t_next, t_str, t_cmp, t_coach, t_data = st.tabs(
    [":material/space_dashboard: Overview", ":material/trending_up: Lifts", ":material/flag: Plateaus",
     ":material/event_upcoming: Next workout", ":material/emoji_events: Goals", ":material/compare_arrows: Compare",
     ":material/forum: Coach", ":material/tune: Settings"])

# ================================================================== OVERVIEW
with t_over:
    # headline numbers: last 4 weeks, compared with the 4 weeks before, with a 12-week trend line
    _ends = [last_day - pd.Timedelta(days=7 * k) for k in range(11, -1, -1)]

    def _per_week(days):
        d = pd.to_datetime(pd.Series(days))
        return [int(((d > e - pd.Timedelta(days=7)) & (d <= e)).sum()) for e in _ends]

    def _in(days, lo, hi):
        d = pd.to_datetime(pd.Series(days))
        return int(((d > last_day - pd.Timedelta(days=hi)) & (d <= last_day - pd.Timedelta(days=lo))).sum())

    hs_days = A.hard_sets(df)["day"]
    pr_days = prs["Date"] if len(prs) else pd.Series([], dtype="datetime64[ns]")
    sets28, sets_prev = _in(hs_days, 0, 28) / 4, _in(hs_days, 28, 56) / 4
    prs28, prs_prev = _in(pr_days, 0, 28), _in(pr_days, 28, 56)
    has_prev = len(p28) > 0
    kpi_row([
        ("Workouts · 4 wks", f"{len(l28)}", f"{len(sess)} total · aim for 12-20" if len(l28) < 12 else f"{len(sess)} total · solid",
         delta_badge(len(l28) - len(p28)) if has_prev else "", spark(_per_week(sess["day"]))),
        ("Hard sets / week", f"{sets28:.0f}", "average over the last 4 weeks",
         delta_badge(sets28 - sets_prev) if has_prev else "", spark(_per_week(hs_days), "#199e70")),
        ("PRs · 4 wks", f"{prs28}", f"{len(prs)} all time",
         delta_badge(prs28 - prs_prev) if has_prev else "", spark(_per_week(pr_days), "#c98500")),
        ("Stalled lifts", f"{n_stalled}", "need a change of plan" if n_stalled else "nothing stuck", "", ""),
    ])

    # the three findings that matter most right now (worst first), from the same rules as the Coach tab
    order = {"crit": 0, "warn": 1, "good": 2, "info": 3}
    top3 = sorted(review["bullets"], key=lambda b: order.get(b[0], 9))[:3]
    dot = {"crit": "#ff7b7b", "warn": "#f2b33d", "good": "#4cc76a", "info": "#8cc0ff"}
    st.markdown('<div class="insight"><div class="h">Right now</div>' + "".join(
        f'<div class="it"><span class="dot" style="background:{dot.get(k, "#8cc0ff")}"></span><span>{escape(t)}</span></div>'
        for k, t in top3) + "</div>", unsafe_allow_html=True)

    # ---- calendar
    section("Training calendar",
            "One square per day. Blue means you trained, and brighter blue means more total weight moved.")
    guide("Each column is a week, each row a weekday. Hover a square for the workout name and volume.",
          "A regular pattern with few empty weeks. Consistency drives results more than any program.",
          "Look for long empty stretches. Those gaps are your biggest lever, bigger than changing exercises.")
    weeks, z, txt = A.calendar_grid(df)
    # Mondays of the calendar's columns (same range as A.calendar_grid). Kept here, not in analysis.py,
    # because Streamlit Cloud hot-reloads app.py but can keep an older analysis module after a push.
    _d0, _d1 = df["day"].min(), df["day"].max()
    week_starts = list(pd.date_range(_d0 - pd.Timedelta(days=_d0.weekday()), _d1, freq="7D"))
    week_keys = [f"{d:%Y-%m-%d}" for d in week_starts]  # unique even across years
    fig = go.Figure(go.Heatmap(
        z=z, x=week_keys, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], text=txt,
        hovertemplate="%{text}<extra></extra>", xgap=3, ygap=3, hoverongaps=True,
        colorscale=[[0, "#184f95"], [1, "#9ec5f4"]], showscale=False))
    style(fig, 310, legend=False)
    fig.update_layout(plot_bgcolor="#242423", margin=dict(l=12, r=16, t=28, b=40))
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category", tickfont=dict(color=INK2, size=11), showticklabels=True)
    # one label per month (at its first week) so the dates never pile up on narrow screens
    multi_year = week_starts[0].year != week_starts[-1].year
    month_ticks, last_m, last_i = [], None, -99
    for i, (key, d0) in enumerate(zip(week_keys, week_starts)):
        m = (d0 + pd.Timedelta(days=6)).to_period("M")  # the month this week mostly ends in
        if m != last_m:
            label = m.strftime("%b %y") if (multi_year and m.month == 1) else m.strftime("%b")
            if month_ticks and i - last_i < 3:  # too close to the previous label: keep the newer month
                month_ticks.pop()
            month_ticks.append((key, label))
            last_m, last_i = m, i
    fig.update_xaxes(type="category", tickangle=0, side="top",
                     tickvals=[k for k, _ in month_ticks], ticktext=[t for _, t in month_ticks])
    show(fig)

    # ---- balance
    section("Are you training every muscle enough?",
            "Hard sets per week for each muscle over your last 4 weeks. A hard set is any working set, "
            "so warm-ups are excluded.")
    guide("Each bar is a muscle. The shaded band is 10-20 sets per week, a commonly used range for building muscle.",
          "Every muscle you care about inside or near the band.",
          "Below the band, add 2-4 sets for that muscle next week. Above it, you may be doing more than you can recover from.")
    fig = go.Figure()
    fig.add_vrect(x0=10, x1=20, fillcolor=BLUE, opacity=0.10, line_width=0)
    for _, r in bal.iterrows():
        fig.add_bar(y=[r["muscle"]], x=[r["sets_per_week"]], orientation="h", name=r["muscle"],
                    marker=dict(color=A.MUSCLE_COLORS[r["muscle"]], line=dict(color=SURFACE, width=2)),
                    hovertemplate=f"<b>{r['muscle']}</b><br>{r['sets_per_week']:.1f} hard sets/week<br>{r['status']}<extra></extra>",
                    text=[f"{r['sets_per_week']:.1f}"], textposition="outside", textfont=dict(color=INK2))
    style(fig, 330)
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Hard sets per week", gridcolor=GRID, showgrid=True, range=[0, max(24, bal["sets_per_week"].max() * 1.15)])
    show(fig)
    labels = " &nbsp; ".join(
        f"<b>{r['muscle']}</b> " + chip({"Low": "warn", "In range": "good", "High": "info"}[r["status"]], r["status"])
        for _, r in bal.iterrows())
    st.markdown(f'<div style="line-height:2.3">{labels}</div>', unsafe_allow_html=True)

    # ---- per-lift volume
    section("Which lifts are you neglecting?",
            "Hard sets for each lift in each of the last 8 weeks, plus the lifts you used to do and dropped.")
    guide("Each row is a lift and each column a week. Lighter blue means more hard sets that week. Empty squares are weeks you skipped it.",
          "Your main lifts show up almost every week with a steady number of sets.",
          "Dropped or rare lifts you still care about? Put them back into your next two sessions.")
    mat = F.lift_weekly_matrix(df, 8, 12)
    if len(mat):
        fig = go.Figure(go.Heatmap(
            z=mat.values, x=list(mat.columns), y=list(mat.index), xgap=3, ygap=3, showscale=False,
            colorscale=[[0, "#242423"], [0.0001, "#1f5fae"], [1, "#4a90e2"]],
            text=np.where(mat.values == 0, "", mat.values.astype(str)), texttemplate="%{text}",
            textfont=dict(color=INK, size=11),
            hovertemplate="<b>%{y}</b><br>week ending %{x}<br>%{z} hard sets<extra></extra>"))
        style(fig, max(260, 30 * len(mat) + 90), legend=False)
        fig.update_layout(plot_bgcolor="#242423")
        fig.update_yaxes(autorange="reversed", type="category", showgrid=False, tickfont=dict(color=INK2, size=11))
        fig.update_xaxes(type="category", side="top", showgrid=False)
        show(fig)
    lv = F.lift_volume(df)
    flagged = lv[lv["Status"].isin(["Dropped", "Rare"])] if len(lv) else lv
    if len(flagged):
        lines = []
        for _, r in flagged.iterrows():
            note = (f"last done {r['Last done (days ago)']} days ago" if r["Status"] == "Dropped"
                    else "only 1 session in the last 4 weeks")
            lines.append(f'{chip("crit" if r["Status"] == "Dropped" else "warn", r["Status"])} '
                         f'<b>{escape(r["Exercise"])}</b> <span style="color:#a9a89f">· {note}</span>')
        # the list can run to 20+ lifts; show the first few and fold the rest
        st.markdown('<div style="line-height:2.2">' + "<br>".join(lines[:5]) + "</div>", unsafe_allow_html=True)
        if len(lines) > 5:
            with st.expander(f"Show {len(lines) - 5} more dropped or rare lifts"):
                st.markdown('<div style="line-height:2.2">' + "<br>".join(lines[5:]) + "</div>", unsafe_allow_html=True)
    if len(lv):
        with st.expander("All lifts: hard sets per week (last 4 weeks)"):
            st.dataframe(lv.style.format({"Sets / week": "{:.1f}"}), width="stretch", hide_index=True)

    # ---- weekly stacked volume
    section("Where your work went, week by week",
            "Every stacked segment is one muscle. Taller bars mean more hard sets that week.")
    guide("Hover a segment to see the exact set count. Colors always map to the same muscle.",
          "Steady bar heights with a mix of colors.",
          "Tall-then-empty weeks show an on-off pattern. Aim for similar volume each week.")
    wk = A.weekly_sets_by_muscle(df)
    fig = go.Figure()
    for m in [m for m in A.MUSCLE_ORDER if m in wk.columns]:
        fig.add_bar(x=wk.index, y=wk[m], name=m, marker=dict(color=A.MUSCLE_COLORS[m], line=dict(color=SURFACE, width=2)),
                    hovertemplate="<b>" + m + "</b><br>Week of %{x|%d %b}<br>%{y} sets<extra></extra>")
    style(fig, 360)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Hard sets")
    show(fig)

    # ---- recent sessions
    section("Recent workouts", "Your latest sessions with total volume (weight × reps summed over working sets).")
    ss = A.session_summary(df).head(8).copy()
    ss["Date"] = ss["Date"].dt.strftime("%d %b %Y")
    ss = ss.rename(columns={"Volume (kg)": f"Volume ({unit})"})
    st.dataframe(ss, width="stretch", hide_index=True)

# ================================================================== LIFT PROGRESS
with t_lift:
    section("Lift progress",
            "Pick one exercise to see if it is getting stronger. This is the most useful page once you have 4+ sessions of a lift.")
    guide("The blue line is your estimated 1-rep max (e1RM) from your best set each day. Orange is the heaviest weight you used. Gold dots are new PRs.",
          "A line that trends up over weeks, with the odd PR dot.",
          "Flat for 4+ weeks? See the Plateaus tab for what to change.")
    pick = st.selectbox("Exercise", list(lifts.index), format_func=lambda k: f"{k.strip()}  ·  {lifts[k]} sessions")
    e = A.e1rm_with_prs(df, pick)
    best, cur = e["e1rm"].max(), e["e1rm"].iloc[-1]
    chg = (cur / e["e1rm"].iloc[0] - 1) * 100 if len(e) > 1 else 0
    kpi_row([
        ("Best e1RM", f"{best:.0f} {unit}", "your strongest estimated max"),
        ("Latest e1RM", f"{cur:.0f} {unit}", f"{e['day'].iloc[-1]:%d %b %Y}"),
        ("Change since first", f"{chg:+.0f}%", f"over {len(e)} sessions"),
        ("PRs on this lift", f"{int(e['is_pr'].sum())}", "times you beat your best"),
    ])

    fig = go.Figure()
    fig.add_scatter(x=e["day"], y=e["e1rm"], name="Estimated 1RM", mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=8, color=BLUE, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{x|%d %b %Y}<br>e1RM <b>%{y:.1f}</b><extra></extra>")
    fig.add_scatter(x=e["day"], y=e["top_weight"], name="Heaviest weight used", mode="lines+markers",
                    line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=7, color=ORANGE, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{x|%d %b %Y}<br>Top weight <b>%{y:.1f}</b><extra></extra>")
    p = e[e["is_pr"]]
    if len(p):
        fig.add_scatter(x=p["day"], y=p["e1rm"], name="New PR", mode="markers",
                        marker=dict(size=14, color="#fab219", symbol="star", line=dict(color=SURFACE, width=2)),
                        hovertemplate="%{x|%d %b %Y}<br>PR: <b>%{y:.1f}</b><extra></extra>")
    style(fig, 380)
    fig.update_layout(hovermode="x unified")  # one tap shows every line's value for that day
    fig.update_yaxes(title=unit)
    show(fig)

    section("Volume per session", "Total weight × reps for this lift. Rising volume at the same weight is also progress.")
    fig = go.Figure(go.Bar(x=e["day"], y=e["volume"], width=86400000 * 2.5, marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                           hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} volume<extra></extra>"))
    style(fig, 260, legend=False)
    show(fig)
    st.caption("e1RM uses the Epley formula (weight × (1 + reps ÷ 30)) on your best working set. "
               "Sets above 12 reps are skipped because the estimate gets unreliable there.")

    section("Rep records", "The heaviest weight you have lifted for each rep count on this lift.")
    guide("Each bar is a rep count (1 rep, 2 reps, ...). Its height is the heaviest weight you have used for that many reps. Counts you never logged are skipped.",
          "Bars that step down smoothly as the rep count goes up.",
          "A bar that is no higher than a bar to its right is an easy record to chase: you already lifted that weight for more reps.")
    rr = F.rep_records(df, pick)
    if len(rr):
        fig = go.Figure(go.Bar(x=[f"{r} rep{'s' if r > 1 else ''}" for r in rr["Reps"]], y=rr["Best weight"],
                               marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                               customdata=rr["Date"].dt.strftime("%d %b %Y"),
                               hovertemplate="%{x}<br><b>%{y:g}</b> on %{customdata}<extra></extra>",
                               text=[f"{v:g}" for v in rr["Best weight"]], textposition="outside", textfont=dict(color=INK2)))
        style(fig, 300, legend=False)
        fig.update_yaxes(title=unit, range=[0, rr["Best weight"].max() * 1.18])
        show(fig)
        easy = [int(r["Reps"]) for _, r in rr.iterrows()
                if (rr.loc[rr["Reps"] > r["Reps"], "Best weight"] >= r["Best weight"]).any()]
        if easy:
            st.markdown(chip("info", "Easy records to chase") + " &nbsp; " +
                        ", ".join(f"<b>{r} reps</b>" for r in easy) +
                        ' <span style="color:#a9a89f">· you have lifted this weight for more reps already.</span>',
                        unsafe_allow_html=True)
    else:
        st.info("No usable working sets for this lift.")

# ================================================================== PLATEAUS & PRs
with t_plat:
    section("Do you need a deload?",
            "A deload is an easier week that lets fatigue drop. This checks your last 8 weeks for signs you need one.")
    guide("The bars are hard sets in each rolling week. The chip adds up two signals: how many lifts got weaker lately, and how long you have trained without a lighter week.",
          "Fine: strength flat or rising, with the odd lighter week.",
          "Deload suggested: next week use about 60% of your usual weight (or half the sets), then go back to normal.")
    dkind = {"Deload suggested": "crit", "Watch": "warn", "Fine": "good"}.get(deload["status"], "info")
    st.markdown(chip(dkind, deload["status"]) + "".join(
        f'<div style="color:#c3c2b7;font-size:.9rem;margin-top:6px">• {escape(r)}</div>' for r in deload["reasons"]),
        unsafe_allow_html=True)
    if deload["compared"] >= 3 and deload["down"]:
        st.caption("Lifts that dipped: " + ", ".join(deload["down"][:8]))
    if len(deload["weeks"]):
        wkd = deload["weeks"]
        med = float(wkd.loc[wkd["sets"] > 0, "sets"].median()) if (wkd["sets"] > 0).any() else 0
        fig = go.Figure(go.Bar(x=[f"{d:%d %b}" for d in wkd["window_end"]], y=wkd["sets"], width=0.55,
                               marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                               hovertemplate="week ending %{x}<br>%{y} hard sets<extra></extra>"))
        fig.add_hline(y=med, line=dict(color=MUTED, dash="dot", width=1.5),
                      annotation_text="typical week", annotation_font=dict(color=MUTED, size=11))
        style(fig, 250, legend=False)
        fig.update_yaxes(title="Hard sets")
        fig.update_xaxes(type="category", title="Week ending")
        show(fig)

    section("Which lifts are stuck?",
            "A lift is stalled when your last 3 sessions did not beat the best you did before them. Only lifts with 4+ sessions are checked.")
    guide("Each card is one lift. Red = stalled, green = recent PR, blue = your last PR was long ago, so log it more often to judge.",
          "Mostly green. A stall now and then is normal.",
          "For a stalled lift, try one change at a time (ideas below), then give it 3 sessions.")
    if plateaus.empty:
        st.info("Not enough repeated sessions yet. Log each lift at least 4 times.")
    else:
        cols = st.columns(2)
        for i, (_, r) in enumerate(plateaus.iterrows()):
            stalled = r["Status"] == "Stalled"
            with cols[i % 2]:
                st.markdown(
                    f'<div class="card"><h4>{escape(r["Exercise"].strip())} &nbsp;'
                    f'{chip("crit", "Stalled") if stalled else (chip("good", "Progressing") if r["Status"] == "Progressing" else chip("info", "Needs more data"))}</h4>'
                    f'<div class="row">Best e1RM <b>{r["Best e1RM"]:.0f} {unit}</b> · recent best <b>{r["Recent best"]:.0f} {unit}</b> · '
                    f'{r["Sessions"]} sessions · last PR <b>{r["Days since last PR"]} days ago</b></div></div>',
                    unsafe_allow_html=True)
        with st.expander("What to try when a lift stalls"):
            st.markdown(
                "- **Check recovery first.** Sleep, food, and whether you trained it often enough (once a week is thin).\n"
                "- **Change the rep range.** Move from 8-12 to 5-8 for 3-4 weeks, then come back.\n"
                "- **Add a back-off set.** After your top set, drop 10-15% and do one more set for reps.\n"
                "- **Take a short deload.** One week at about 60% of your usual weight, then retry.\n"
                "- **Swap the variation.** Same muscle, a different angle or machine, for a few weeks.")

    section("How much has each lift changed?", "Percent change in estimated 1RM from your first session of a lift to your latest.")
    guide("Bars to the right are strength gained. Hover for first and latest values.",
          "Most bars pointing right.", "Bars at zero or left are lifts to look at first.")
    ch = A.lift_change(df)
    if len(ch):
        fig = go.Figure(go.Bar(
            y=ch["Exercise"], x=ch["Change %"], orientation="h",
            marker=dict(color=np.where(ch["Change %"] >= 0, BLUE, "#e66767"), line=dict(color=SURFACE, width=2)),
            customdata=np.stack([ch["First"], ch["Latest"], ch["Sessions"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{x:+.0f}%<br>first e1RM %{customdata[0]:.0f} → latest %{customdata[1]:.0f}"
                          "<br>%{customdata[2]} sessions<extra></extra>",
            text=[f"{v:+.0f}%" for v in ch["Change %"]], textposition="outside", textfont=dict(color=INK2)))
        style(fig, max(260, 34 * len(ch) + 80), legend=False)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(title="e1RM change (%)", showgrid=True, gridcolor=GRID,
                         range=[min(0, ch["Change %"].min()) * 1.9 - 10, ch["Change %"].max() * 1.22 + 5])
        show(fig)

    section("PR timeline", "Every time a lift's estimated max beat everything before it. The first session of a lift is a baseline, not a PR.")
    if len(plateaus):
        dr = plateaus.sort_values("Days since last PR", ascending=False).head(5)
        st.markdown("**Longest since a PR:** " + " &nbsp; ".join(
            f'{chip("crit" if r["Status"] == "Stalled" else "info", escape(r["Exercise"].strip()))} {r["Days since last PR"]} d'
            for _, r in dr.iterrows()), unsafe_allow_html=True)
    if len(prs):
        t = F.pr_feed(df)
        t["Date"] = t["Date"].dt.strftime("%d %b %Y")
        t["Exercise"] = t["Exercise"].str.strip()
        t["Gain"] = t["Gain"].round(1)
        t["Gain %"] = t["Gain %"].round(1)
        t = t.rename(columns={"e1RM": f"New e1RM ({unit})", "Prev best": f"Old best ({unit})",
                              "Top weight": f"Top weight ({unit})", "Gain": f"Gain ({unit})"})
        st.dataframe(t, width="stretch", hide_index=True)
    else:
        st.write("No PRs yet.")

# ================================================================== NEXT WORKOUT
with t_next:
    section("Your next full workout",
            "Repcompass guesses which workout comes next from your usual rotation, then sets a target for every lift in it.")
    guide("The title is the workout you most likely do next. Each row shows the number of sets and the weight × reps to aim for.",
          "You can follow the whole table without thinking.",
          "Disagree with the guess? Pick another workout below. Targets always use your last session of each lift.")
    titles_all = sorted(df["Title"].unique())
    plan0 = F.plan_next_session(df, int(low), int(high), float(step), plateaus)
    if plan0:
        pick_title = st.selectbox("Workout", titles_all, index=titles_all.index(plan0["title"]), key="plan_title")
        plan = plan0 if pick_title == plan0["title"] else F.plan_next_session(
            df, int(low), int(high), float(step), plateaus, title=pick_title)
        if len(plan["table"]):
            body = "".join(
                f'<tr><td class="l">{escape(r["Exercise"])}{" " + chip("crit", "Stalled") if r["Stalled"] else ""}</td>'
                f'<td class="s">{r["Sets"]}</td><td class="t">{escape(r["Target"])}</td><td class="n">{escape(r["Note"])}</td></tr>'
                for _, r in plan["table"].iterrows())
            st.markdown(
                f'<div class="card"><h4>{escape(plan["title"])} · next workout</h4>'
                f'<div class="row" style="margin:0 0 6px 0">{escape(plan["how"])} Based on your last one on {plan["based_on"]:%d %b %Y}.</div>'
                f'<table class="plan"><tr><th>Lift</th><th>Sets</th><th>Target ({unit} × reps)</th><th>Why</th></tr>{body}</table></div>',
                unsafe_allow_html=True)
        else:
            st.info("No loaded lifts found in that workout.")

    section("Or pick lifts one by one",
            "A target for each lift based on what you did last time. This uses double progression: fill the rep range first, then add weight.")
    guide(f"Each card shows last session, then the weight and reps to aim for. Your rep range is {low}-{high} (change it in Settings).",
          "Hitting the target reps on every set.",
          f"Hit {high} reps on every set → the card tells you to add {step:g} {unit}. Missed it → repeat the weight and aim for one more rep.")
    ex_list = list(lifts.index)
    chosen = st.multiselect("Lifts for your next session", ex_list, default=ex_list[:6],
                            format_func=lambda k: k.strip())
    cols = st.columns(3)
    shown = 0
    for ex in chosen:
        s = A.suggest_next(df, ex, int(low), int(high), float(step))
        if not s:
            continue
        up_ = s["Next weight (kg)"] > 0 and "add weight" in s["Why"].lower()
        with cols[shown % 3]:
            st.markdown(
                f'<div class="card" style="min-height:250px"><h4>{escape(ex.strip())}</h4>'
                f'<div class="big">{s["Next weight (kg)"]:g} {unit} × {s["Target reps per set"]}</div>'
                f'<div class="row">{chip("good", "Add weight") if up_ else chip("info", "Hold weight")}</div>'
                f'<div class="row">Last time ({s["Last session"]:%d %b}): {escape(s["Last working sets"])}</div>'
                f'<div class="why">{escape(s["Why"])}</div></div>', unsafe_allow_html=True)
        shown += 1
    st.caption("These are suggestions, not rules. If a set felt very easy or very hard, trust your body over the number.")

# ================================================================== BODYWEIGHT (bottom of Lifts)
with t_lift:
    section("Bodyweight & timed work", "Pull-ups, dips with no added weight, and holds like planks. These do not have a weight, so we track reps and seconds.")
    guide("Each line is one exercise. Higher is better.",
          "Best reps climbing slowly over months.",
          "Stuck? Add a rep a week, slow the lowering, or add weight once you pass about 12 reps.")
    bw = A.bodyweight_progress(df)
    tp = A.timed_progress(df)
    if len(bw):
        fig = go.Figure()
        for i, (ex, g) in enumerate(bw.groupby("Exercise")):
            fig.add_scatter(x=g["day"], y=g["best_reps"], name=ex.strip(), mode="lines+markers",
                            line=dict(color=A.SERIES[i % 8], width=2),
                            marker=dict(size=8, color=A.SERIES[i % 8], line=dict(color=SURFACE, width=2)),
                            hovertemplate="%{x|%d %b %Y}<br><b>%{y:.0f}</b> reps (best set)<extra>" + ex.strip() + "</extra>")
        style(fig, 340)
        fig.update_yaxes(title="Best reps in one set")
        show(fig)
    if len(tp):
        fig = go.Figure()
        for i, (ex, g) in enumerate(tp.groupby("Exercise")):
            fig.add_scatter(x=g["day"], y=g["best_seconds"], name=ex.strip(), mode="lines+markers",
                            line=dict(color=A.SERIES[(i + 2) % 8], width=2),
                            marker=dict(size=8, color=A.SERIES[(i + 2) % 8], line=dict(color=SURFACE, width=2)),
                            hovertemplate="%{x|%d %b %Y}<br><b>%{y:.0f}</b> seconds<extra>" + ex.strip() + "</extra>")
        style(fig, 300)
        fig.update_yaxes(title="Seconds held")
        show(fig)
    if not len(bw) and not len(tp):
        st.info("No bodyweight or timed exercises found in this export.")

# ================================================================== DATA & SETTINGS
with t_data:
    section("Training settings", "These shape the targets in Next workout and the units shown everywhere.")
    sc1, sc2 = st.columns(2, gap="medium")
    with sc1.container(border=True):
        st.radio("Units in your export", ["kg", "lb"], horizontal=True, key="unit", on_change=_unit_changed,
                 help="Lyfta exports numbers in whatever unit your app is set to. Pick the same one here.")
        st.number_input(f"Weight jump ({unit})", 0.5, 20.0, step=0.5, key="step", help="The smallest plate jump you use.")
    with sc2.container(border=True):
        rc = st.columns(2)
        rc[0].number_input("Rep range: low", 3, 20, key="low", help="Below this many reps, hold the weight and build reps first.")
        rc[1].number_input("Rep range: high", 4, 30, key="high", help="Hit this on every set and the app tells you to add weight.")
        st.caption("Your export has no effort score (RPE), so targets use weight and reps only.")

    section("Your data", "What is loaded right now. Repcompass never stores your files on a server.")
    if use_demo:
        st.info("You are looking at demo data.")
        st.button("Leave demo", on_click=stop_demo, icon=":material/logout:")
    else:
        rows = "".join(
            f'<div class="it" style="display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid var(--line)">'
            f'<span><b>{escape(str(info["name"]))}</b><br><span style="color:var(--muted);font-size:.82rem">{escape(str(info["source"]))}</span></span>'
            f'<span style="color:var(--ink2);font-size:.85rem;white-space:nowrap">{info["rows"]} sets</span></div>'
            for info in infos)
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
        if raw_export is not None:
            label = "Download combined history (CSV)" if len(infos) > 1 else "Download a clean copy (CSV)"
            st.download_button(label, raw_export.to_csv(index=False).encode("utf-8"), icon=":material/download:",
                               file_name="repcompass_history.csv", mime="text/csv")
    st.toggle("Remember my data on this device", key="remember",
              help="Keeps your combined history and settings in this browser (not on any server), so next visit "
                   "opens straight to your dashboard. New exports are added on top. Turn off on a shared computer.")
    st.caption("To move to another phone or computer: download the combined history above and load it there.")

    with st.expander("Advanced: sync with a Lyfta API key (optional)", icon=":material/key:",
                     expanded=bool(st.session_state.get("lyfta_err"))):
        st.caption(f"Most people should use the CSV export, it needs nothing extra. If you can log in at "
                   f"[my.lyfta.app]({L.KEY_PAGE}) on the web, you can generate a personal key there and paste it here "
                   "to pull workouts without exporting. It only reads your workouts.")
        st.text_input("Lyfta API key", type="password", key="lyfta_key")
        st.checkbox("Remember key on this device", key="remember_key",
                    help="Saved in this browser only, so the app syncs by itself next visit. Leave off on a shared computer.")
        if st.button("Sync now", type="primary", disabled=not lyfta_key, icon=":material/sync:"):
            run_sync(lyfta_key)
            st.rerun()
        if st.session_state.get("lyfta_err"):
            st.error(st.session_state["lyfta_err"])
        elif st.session_state.get("lyfta_sync"):
            _f, _i = st.session_state["lyfta_sync"]
            st.caption(f"Synced {_i['workouts']} workouts at {st.session_state.get('lyfta_sync_at')}.")
        for info in infos:
            if info.get("unknown_set_types"):
                st.caption(f"Lyfta sent set types Repcompass does not know yet ({', '.join(info['unknown_set_types'])}). "
                           "They are counted as normal sets.")

    section("Save and restore your setup",
            "Your goals, muscle fixes, bodyweight and settings. With Remember on they are kept in this browser automatically. "
            "Download them to move to another device.")
    setup = current_setup()
    cc = st.columns(2)
    cc[0].download_button("Download my setup (JSON)", json.dumps(setup, indent=2).encode("utf-8"),
                          file_name="repcompass_setup.json", mime="application/json")
    up_setup = cc[1].file_uploader("Restore a setup file (JSON)", type="json", key="setup_up")
    if up_setup is not None and st.session_state.get("_setup_done") != up_setup.file_id:
        try:
            loaded = json.loads(up_setup.getvalue().decode("utf-8"))
            if loaded.get("app") != "repcompass":
                raise ValueError("not a Repcompass setup file")
            apply_setup(loaded)
            st.session_state["_setup_done"] = up_setup.file_id
            st.rerun()
        except Exception:
            st.error("That file is not a valid Repcompass setup.")

    section("Fix muscle groups",
            "Repcompass guesses each exercise's muscle from its name. If one is wrong, change it here and every chart updates.")
    guide("Each row is an exercise. The Muscle column is a dropdown.",
          "Every exercise sits under the muscle you actually train with it.",
          "Fix the exercises with the most sets first. Cardio and Other are left out of the muscle charts.")
    em = F.exercise_muscles(df)
    edited = st.data_editor(
        em, hide_index=True, width="stretch", key="muscle_editor", disabled=["Exercise", "Sets"],
        column_config={"Muscle": st.column_config.SelectboxColumn("Muscle", options=F.MUSCLE_CHOICES, required=True)})
    new_ov = {r["Exercise"]: r["Muscle"] for _, r in edited.iterrows() if r["Muscle"] != A.muscle_group(r["Exercise"])}
    if new_ov != overrides:
        st.session_state["muscle_overrides"] = new_ov
        st.rerun()
    if overrides and st.button("Reset all muscle fixes"):
        st.session_state["muscle_overrides"] = {}
        st.session_state.pop("muscle_editor", None)
        st.rerun()

    section("Where your data lives", "A quick note on how this app handles your data.")
    st.markdown(
        "- Exports (and optional Lyfta API syncs) are read in memory to draw the charts. Nothing is saved on a server and there are no accounts.\n"
        "- With **Remember my data on this device** on, your combined history and settings are kept in this browser's "
        "local storage. Another browser or device starts empty: load the history CSV you downloaded above.\n"
        "- A Lyfta key is only sent to Lyfta, through this app, to read your workouts. It is kept in this browser only if you tick Remember key.\n"
        "- Running the app locally keeps everything on your own computer.")
    st.button("Forget this device", on_click=forget_device, icon=":material/delete:",
              help="Deletes the saved history, settings and Lyfta key from this browser.")


# ================================================================== GLOSSARY (bottom of Settings)
with t_data:
    section("Plain-English guide", "Every term used in this app, and the assumptions behind the numbers.")
    terms = [
        ("Estimated 1RM (e1RM)", "A guess at the heaviest single rep you could do, worked out from a set with more reps. Formula: weight × (1 + reps ÷ 30). It lets you compare sessions even when the reps differ. Sets above 12 reps are ignored."),
        ("Hard set", "Any set that is not a warm-up and has reps logged. Counting hard sets per muscle is a simple way to check you train everything enough."),
        ("Volume", "Weight × reps added up. 3 sets of 10 reps at 50 is 1,500 volume. Used to compare sessions."),
        ("PR (personal record)", "A session where a lift's e1RM beat everything you did before it."),
        ("Plateau / stalled", "The last 3 sessions of a lift did not beat the best before them."),
        ("Double progression", "First fill your rep range at one weight. Once every set hits the top of the range, add weight and start at the bottom again."),
        ("Deload", "An easier week (about 60% of your usual weight, or half the sets) so fatigue can drop. Repcompass suggests one when several lifts get weaker and you have trained hard for many weeks in a row."),
        ("Strength level", "Your estimated 1RM divided by your bodyweight, compared with rough Beginner to Elite ratios. A guideline only: it ignores age, height and how long you have trained."),
        ("Goal ETA", "A straight line fitted to your estimated 1RM over the last 4 months, extended to your target. Real progress slows down, so treat the date as a best case."),
        ("Rep record", "The heaviest weight you have used for an exact number of reps, such as your best 5-rep weight."),
        ("RPE", "How close to failure a set was, from 1-10. Your Lyfta export does not include it, so the app cannot use it."),
    ]
    for term, d in terms:
        st.markdown(f'<div class="card"><h4>{term}</h4><div class="row" style="margin-top:0;color:#c3c2b7">{d}</div></div>', unsafe_allow_html=True)
    section("Good to know", "Honest limits so you do not over-trust the numbers.")
    st.markdown(
        "- Muscle groups are guessed from exercise names, so an unusual exercise may land in the wrong group.\n"
        "- The 10-20 sets per week band is a general guideline, not a prescription for everyone.\n"
        "- Few sessions means shaky conclusions. Plateau checks need 4+ sessions of the same lift.\n"
        "- Left and right sets of one-sided exercises are counted as separate sets.\n"
        "- If you run this locally, your data stays on your computer.")


# ================================================================== STRENGTH & GOALS
with t_str:
    section("Strength levels",
            "How your bench, squat, deadlift and overhead press compare with common bodyweight standards.")
    guide("Each card is one lift. The bar shows how far you are from the next level.",
          "Intermediate or better on the lifts you train most.",
          "Chase the lift with the shortest bar first. Beginners often level up fastest on their weakest lift.")
    c1, c2 = st.columns(2)
    st.session_state.setdefault("bw", 70.0 if unit == "kg" else 155.0)  # no value= so a restored setup never clashes
    bwt = c1.number_input(f"Your bodyweight ({unit})", 30.0, 300.0, step=0.5, key="bw")
    sex = c2.radio("Standards table", ["Male", "Female"], horizontal=True, key="sex",
                   help="The tables differ because average strength ratios differ. Pick the one you want to compare against.")
    levels = F.strength_levels(df, float(bwt), sex)
    if levels.empty:
        st.info("No barbell bench press, squat, deadlift or overhead press found in your lifts.")
    else:
        cols = st.columns(2)
        for i, (_, r) in enumerate(levels.iterrows()):
            nxt = (f"Next: <b>{r['Next level']}</b> at {r['Needed e1RM']:.0f} {unit} "
                   f"({r['Needed e1RM'] - r['Best e1RM']:.0f} {unit} to go)") if r["Next level"] != "Top level" else "Top level reached."
            kind = "info" if r["Level"] in ("Untrained", "Beginner", "Novice") else "good"
            with cols[i % 2]:
                st.markdown(
                    f'<div class="card"><h4>{r["Lift"]} &nbsp; {chip(kind, r["Level"])}</h4>'
                    f'<div class="big">{r["x Bodyweight"]:.2f}× bodyweight</div>'
                    f'<div class="row">Best e1RM <b>{r["Best e1RM"]:.0f} {unit}</b> from {escape(r["Exercise"])}</div>'
                    f'<div class="bar"><div style="width:{r["Progress"] * 100:.0f}%"></div></div>'
                    f'<div class="why">{nxt}</div></div>', unsafe_allow_html=True)
    st.caption("Rough guidelines from commonly used bodyweight ratios. They ignore age, height and training history. "
               "Only barbell versions are matched; dumbbell and machine variants are skipped. Uses your best e1RM from the last 4 months.")

    section("Goals", "Set a target estimated 1RM for a lift and see when you will reach it at your current pace.")
    guide("Each card shows how close you are and a projected date. The date comes from a straight line through your last 4 months of e1RM.",
          "A projected date before your deadline, shown as On track.",
          "Behind? The card shows the weekly gain you would need. Hitting it usually means more frequent training, not harder single sessions.")
    goals = st.session_state.setdefault("goals", [])
    with st.form("goal_form", clear_on_submit=True):
        gc = st.columns([3, 1.4, 1.8])
        gx = gc[0].selectbox("Lift", list(lifts.index), format_func=lambda k: k.strip())
        gt = gc[1].number_input(f"Target e1RM ({unit})", 1.0, 1000.0, 100.0, step=2.5)
        gd = gc[2].date_input("By when (optional)", value=None, min_value=date.today())
        if st.form_submit_button("Add goal"):
            goals.append({"exercise": gx, "target": float(gt), "date": gd.isoformat() if gd else None})
    st.caption("A goal of '100 kg for 1 rep' is a target e1RM of 100. For a rep target, use weight × (1 + reps ÷ 30).")
    for i, g in enumerate(list(goals)):
        p = F.goal_projection(df, g["exercise"], g["target"], g.get("date"))
        if not p.get("ok"):
            continue
        if p["reached"]:
            status = chip("good", "Reached")
        elif p["eta"] is not None:
            status = f'Projected <b>{p["eta"]:%d %b %Y}</b> at +{p["slope_week"]:.1f} {unit}/week'
            if p["on_track"] is True:
                status += " &nbsp; " + chip("good", "On track")
            elif p["on_track"] is False:
                status += " &nbsp; " + chip("warn", f"Behind: needs +{p['needed_week']:.1f} {unit}/week")
        else:
            status = escape(p["note"])
        deadline = f" · by {date.fromisoformat(g['date']):%d %b %Y}" if g.get("date") else ""
        gcol = st.columns([12, 1.4])
        gcol[0].markdown(
            f'<div class="card"><h4>{escape(g["exercise"].strip())}{deadline}</h4>'
            f'<div class="big">{p["current"]:.0f} / {g["target"]:g} {unit}</div>'
            f'<div class="bar"><div style="width:{p["progress"] * 100:.0f}%"></div></div>'
            f'<div class="row">{status}</div></div>', unsafe_allow_html=True)
        if gcol[1].button("Remove", key=f"rm_goal_{i}"):
            goals.pop(i)
            st.rerun()


# ================================================================== COMPARE
with t_cmp:
    section("Compare two periods", "Pick any two date ranges and see what changed: how often you trained, how much, and which lifts moved.")
    guide("Period A is the earlier range and B the later one. The cards show B, with the change against A underneath.",
          "More sets and PRs in B without a drop in your lifts.",
          "Strength down while volume is way up? That is a sign of fatigue. See the deload check on the Plateaus tab.")
    d_min, d_max = df["day"].min().date(), df["day"].max().date()
    b_start = max(d_max - timedelta(days=27), d_min)
    a_end = max(b_start - timedelta(days=1), d_min)
    a_start = max(a_end - timedelta(days=27), d_min)
    cc = st.columns(2)
    ra = cc[0].date_input("Period A (earlier)", value=(a_start, a_end), min_value=d_min, max_value=d_max, key="cmp_a")
    rb = cc[1].date_input("Period B (later)", value=(b_start, d_max), min_value=d_min, max_value=d_max, key="cmp_b")
    if len(ra) == 2 and len(rb) == 2:
        cmp_ = F.compare_periods(df, ra, rb)
        a_, b_ = cmp_["a"], cmp_["b"]

        def delta(key, fmt="{:+,.0f}"):
            return f"{fmt.format(b_[key] - a_[key])} vs A ({a_[key]:,.0f})" if key != "per_week" else \
                f"{b_[key] - a_[key]:+.1f} vs A ({a_[key]:.1f})"

        kpi_row([
            ("Workouts", f"{b_['sessions']}", delta("sessions")),
            ("Hard sets", f"{b_['hard_sets']}", delta("hard_sets")),
            (f"Volume ({unit})", f"{b_['volume']:,.0f}", delta("volume")),
            ("New PRs", f"{b_['prs']}", delta("prs")),
        ])
        if cmp_["muscles"].shape[0]:
            section("Hard sets per muscle", "Grey is period A, blue is period B.")
            mu = cmp_["muscles"]
            fig = go.Figure()
            fig.add_bar(x=mu.index, y=mu["A"], name="Period A", marker=dict(color="#6b6a63", line=dict(color=SURFACE, width=2)),
                        hovertemplate="%{x}<br>A: <b>%{y:.0f}</b> sets<extra></extra>")
            fig.add_bar(x=mu.index, y=mu["B"], name="Period B", marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                        hovertemplate="%{x}<br>B: <b>%{y:.0f}</b> sets<extra></extra>")
            style(fig, 320)
            fig.update_layout(barmode="group")
            fig.update_yaxes(title="Hard sets")
            show(fig)
        lf = cmp_["lifts"]
        if len(lf):
            section("Lift changes", "Best estimated 1RM in B compared with A, for lifts you did in both.")
            fig = go.Figure(go.Bar(
                y=lf["Exercise"], x=lf["Change %"], orientation="h",
                marker=dict(color=np.where(lf["Change %"] >= 0, BLUE, "#e66767"), line=dict(color=SURFACE, width=2)),
                customdata=np.stack([lf["A"], lf["B"]], axis=-1),
                hovertemplate="<b>%{y}</b><br>%{x:+.1f}%<br>A %{customdata[0]:.0f} → B %{customdata[1]:.0f}<extra></extra>",
                text=[f"{v:+.1f}%" for v in lf["Change %"]], textposition="outside", textfont=dict(color=INK2)))
            style(fig, max(240, 34 * len(lf) + 80), legend=False)
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(title="e1RM change (%)", showgrid=True, gridcolor=GRID,
                             range=[min(0, lf["Change %"].min()) * 1.9 - 4, lf["Change %"].max() * 1.25 + 4])
            show(fig)
        else:
            st.info("No lift appears in both periods with enough data to compare.")
    else:
        st.info("Pick a start and an end date for both periods.")


# ================================================================== COACH & SHARE
@st.cache_data(show_spinner=False)
def _card_png(_df, key, month, unit_, fmt):
    return R.share_card(_df, pd.Period(month), unit_, fmt)


@st.cache_data(show_spinner=False)
def _report_pdf(_df, key, month, unit_, lo, hi, stp, ov):
    return R.monthly_report_pdf(_df, pd.Period(month), unit_, lo, hi, stp)


with t_coach:
    section("Your weekly review", "A plain-English read of your recent training, written from your own numbers.")
    guide("Each line is one finding. Green is going well, yellow is worth a look, red needs a change.",
          "Mostly green lines and a clear plan for the red ones.",
          "Pick the one or two red or yellow lines that matter most and fix only those next week.")
    label = {"good": "Good", "warn": "Look at", "crit": "Fix", "info": "Note"}
    for kind, text in review["bullets"]:
        st.markdown(f'<div class="card" style="padding:10px 16px;margin-bottom:8px">{chip(kind, label[kind])} &nbsp; {escape(text)}</div>',
                    unsafe_allow_html=True)
    with st.expander("Want an AI coach's take? Copy this prompt"):
        st.code(review["prompt"], language=None)
        st.caption("Paste it into ChatGPT or Claude. It holds your computed stats only, not your raw workout file.")

    months = R.months_available(df)
    section("Shareable summary card", "A picture of your month to post. Choose the month and the shape.")
    guide("Pick a month and a format, then download the image.",
          "A month with a few PRs and a steady number of workouts looks great.",
          "Post format (4:5) suits Instagram and X feeds. Story format (9:16) suits Reels covers and Stories.")
    sc = st.columns([2, 2, 3])
    m_card = sc[0].selectbox("Month", months, format_func=lambda m: m.strftime("%B %Y"), key="card_month")
    fmt = sc[1].radio("Format", ["Post (4:5)", "Story (9:16)"], horizontal=True, key="card_fmt")
    f_key = "post" if fmt.startswith("Post") else "story"
    png = _card_png(df, data_key, str(m_card), unit, f_key)
    st.image(png, width=360)
    st.download_button("Download image (PNG)", png, file_name=f"repcompass_{m_card}_{f_key}.png", mime="image/png")

    section("Monthly report (PDF)", "A two-page report: totals, PRs, stalled lifts and your next workout.")
    m_rep = st.selectbox("Month for the report", months, format_func=lambda m: m.strftime("%B %Y"), key="rep_month")
    pdf_bytes = _report_pdf(df, data_key, str(m_rep), unit, int(low), int(high), float(step), tuple(sorted(overrides.items())))
    st.download_button("Download report (PDF)", pdf_bytes, file_name=f"repcompass_report_{m_rep}.pdf", mime="application/pdf")


# ------------------------------------------------------------------ remember settings on this device
if remember and not use_demo:
    BS.put("setup", json.dumps(current_setup(), sort_keys=True, default=str))
elif not remember:
    BS.put("setup", None)
flush_store()
