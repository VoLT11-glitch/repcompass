"""Repcompass: turn a Lyfta export into clear answers about your training.

Run:  python -m streamlit run app.py
"""
import json
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

st.set_page_config(page_title="Repcompass", page_icon="🏋️", layout="wide")

# ------------------------------------------------------------------ look & feel
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#8d8c85"
SURFACE, CARD, GRID = "#1a1a19", "#1c1c1b", "#2c2c2a"
BLUE, ORANGE, AQUA = A.SERIES[0], A.SERIES[1], A.SERIES[2]
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 8% -10%, #1b2a42 0%, rgba(18,18,17,0) 60%), #121211; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { letter-spacing: -0.02em; }
section[data-testid="stSidebar"] { background: #161615; border-right: 1px solid #262624; }

.hero { padding: 6px 0 18px 0; }
.hero .kicker { color: #6da7ec; font-weight: 600; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; }
.hero h1 { font-size: 2.6rem; font-weight: 800; margin: .15rem 0 .35rem 0; line-height: 1.05; }
.hero p { color: #c3c2b7; font-size: 1.02rem; max-width: 760px; margin: 0; }

.kpi { background: #1c1c1b; border: 1px solid #2c2c2a; border-radius: 16px; padding: 16px 18px; height: 100%; }
.kpi .l { color: #8d8c85; font-size: .74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.kpi .v { font-size: 1.95rem; font-weight: 750; line-height: 1.15; margin-top: 4px; }
.kpi .s { color: #a9a89f; font-size: .82rem; margin-top: 2px; }

.sec { margin: 30px 0 4px 0; }
.sec h3 { font-size: 1.28rem; font-weight: 700; margin: 0; }
.sec .what { color: #c3c2b7; font-size: .95rem; margin-top: 3px; }

.guide { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 10px 0 14px 0; }
.guide > div { background: #171716; border: 1px solid #262624; border-radius: 12px; padding: 10px 14px; font-size: .86rem; color: #c3c2b7; }
.guide b { display: block; color: #6da7ec; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 3px; }
@media (max-width: 900px) { .guide { grid-template-columns: 1fr; } }

div[data-testid="stPlotlyChart"] { background: #1c1c1b; border: 1px solid #2c2c2a; border-radius: 16px; padding: 0; overflow: hidden; margin-top: 4px; }

.chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .76rem; font-weight: 600; border: 1px solid; }
.chip.good { color: #54d054; border-color: #1c5f1c; background: #0f2a0f; }
.chip.warn { color: #fab219; border-color: #6b4d09; background: #2a2008; }
.chip.crit { color: #ff8a8a; border-color: #6b2a2a; background: #2a1010; }
.chip.info { color: #86b6ef; border-color: #1f3a5f; background: #0f1b2d; }

.card { background: #1c1c1b; border: 1px solid #2c2c2a; border-radius: 16px; padding: 16px 18px; margin-bottom: 14px; }
.card h4 { margin: 0 0 6px 0; font-size: 1rem; font-weight: 700; }
.card .row { color: #a9a89f; font-size: .82rem; margin-top: 8px; }
.card .big { font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; }
.card .why { color: #c3c2b7; font-size: .84rem; margin-top: 8px; border-top: 1px solid #2c2c2a; padding-top: 8px; }

.step { display:flex; gap:14px; align-items:flex-start; background:#1c1c1b; border:1px solid #2c2c2a; border-radius:16px; padding:16px 18px; margin-bottom:10px; }
.step .n { background:#3987e5; color:#fff; font-weight:800; border-radius:10px; min-width:32px; height:32px; display:flex; align-items:center; justify-content:center; }
.step p { margin:2px 0 0 0; color:#c3c2b7; font-size:.92rem; }

.bar { height: 8px; background: #2c2c2a; border-radius: 6px; overflow: hidden; margin: 10px 0 4px 0; }
.bar > div { height: 100%; background: #3987e5; border-radius: 6px; }
.plan { width: 100%; border-collapse: collapse; font-size: .92rem; margin-top: 6px; }
.plan th { text-align: left; color: #8d8c85; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; padding: 6px 8px; border-bottom: 1px solid #2c2c2a; }
.plan td { padding: 10px 8px; border-bottom: 1px solid #232321; color: #e6e5df; vertical-align: top; }
.plan td.t { font-weight: 800; white-space: nowrap; }
.plan td.n { color: #a9a89f; font-size: .82rem; }
button[data-baseweb="tab"] { font-weight: 600; font-size: .95rem; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def section(title: str, what: str):
    st.markdown(f'<div class="sec"><h3>{title}</h3><div class="what">{what}</div></div>', unsafe_allow_html=True)


def guide(read: str, good: str, act: str):
    """Three small boxes under a heading: how to read it, what good looks like, what to do."""
    st.markdown(
        f'<div class="guide"><div><b>How to read it</b>{read}</div>'
        f'<div><b>What good looks like</b>{good}</div><div><b>What to do</b>{act}</div></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def chip(kind, text):
    icons = {"good": "✓", "warn": "!", "crit": "▼", "info": "•"}
    return f'<span class="chip {kind}">{icons[kind]} {text}</span>'


def style(fig, height=380, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=12, r=16, t=28, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK2, size=12),
        hoverlabel=dict(bgcolor="#252523", bordercolor="#3a3a37", font=dict(color=INK, size=12)),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color=INK2), bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        showlegend=legend, barcornerradius=4, hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED), zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, tickfont=dict(color=MUTED), linecolor="rgba(0,0,0,0)", automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


# ------------------------------------------------------------------ sidebar & data
here = Path(__file__).parent
auto = sorted(here.glob("export*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

with st.sidebar:
    st.markdown("### 🏋️ Repcompass")
    st.caption("Your Lyfta data, explained.")
    st.markdown("**1. Your data**")
    ups = st.file_uploader("Workout export (CSV)", type="csv", accept_multiple_files=True, label_visibility="collapsed",
                           help="Lyfta, Hevy or Strong CSV. Add several files to combine them. A workout that appears in more than one file is only counted once.")
    use_demo = st.toggle("Use demo data", value=False, help="Fake 30-session history so you can explore the app.")
    if not ups and not use_demo and auto:
        st.caption(f"Using the newest file in this folder: `{auto[0].name}`")
    st.markdown("**2. Units in your export**")
    unit = st.radio("Unit", ["kg", "lb"], horizontal=True, label_visibility="collapsed",
                    help="Lyfta exports numbers in whatever unit your app is set to. Pick the same one here.")
    st.markdown("**3. How you progress**")
    low = st.number_input("Rep range: low", 3, 20, 8, help="Below this many reps, hold the weight and build reps first.")
    high = st.number_input("Rep range: high", 4, 30, 12, help="Hit this on every set and the app tells you to add weight.")
    step = st.number_input(f"Weight jump ({unit})", 0.5, 20.0, 2.5 if unit == "kg" else 5.0, step=0.5,
                           help="The smallest plate jump you use.")
    st.caption("Your export has no effort score (RPE), so everything here is based on weight and reps only.")

raw_export, infos, load_err = None, [], None
if use_demo:
    df = A.load_demo()
elif ups or auto:
    try:
        df, raw_export, infos = I.load_files(list(ups) if ups else [auto[0]])
    except Exception as exc:  # unreadable or unknown file
        df, load_err = None, str(exc)
else:
    df = None

with st.sidebar:
    if load_err:
        st.error(load_err)
    for info in infos:
        if info["source"] != "Lyfta":
            st.caption(f"Read **{info['name']}** as a {info['source']} export ({info['rows']} rows).")
        if info.get("unit") and info["unit"] != unit:
            st.warning(f"This {info['source']} export is in {info['unit']}. Switch the unit above to {info['unit']}.")


# ------------------------------------------------------------------ onboarding (no data yet)
def onboarding():
    st.markdown(
        '<div class="hero"><div class="kicker">Repcompass</div><h1>Know if your training is working.</h1>'
        "<p>Upload your Lyfta export and get plain-English answers: are you getting stronger, which lifts "
        "have stalled, are you training every muscle enough, and what weight to try next session.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Get started in 3 steps")
    for n, t, d in [
        ("1", "Export your data from Lyfta", "In the Lyfta app open Settings → Export data and choose CSV. Save the file to your computer. Hevy and Strong CSV exports work too."),
        ("2", "Upload it in the sidebar", "Drag the CSV into the box on the left. Repcompass does not save your file. It is only used to draw your charts for this visit."),
        ("3", "Read the tabs left to right", "Overview tells you the big picture, then dig into each lift, your plateaus, and next session's targets."),
    ]:
        st.markdown(f'<div class="step"><div class="n">{n}</div><div><b>{t}</b><p>{d}</p></div></div>', unsafe_allow_html=True)
    st.info("No export yet? Switch on **Use demo data** in the sidebar to explore with a fake training history.")


if df is None:
    onboarding()
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
lifts = (w[w["e1rm"].notna()].groupby("Exercise")["day"].nunique().sort_values(ascending=False))
plateaus = A.plateau_report(df)
n_stalled = int((plateaus["Status"] == "Stalled").sum()) if len(plateaus) else 0
prs = A.pr_timeline(df)
deload = F.deload_check(df)

st.markdown(
    f'<div class="hero"><div class="kicker">Repcompass</div><h1>Your training, explained.</h1>'
    f"<p>{len(sess)} workouts logged from {first_day:%d %b %Y} to {last_day:%d %b %Y}. "
    f"Read the tabs left to right. Each one starts with what it shows and ends with what to do.</p></div>",
    unsafe_allow_html=True,
)

t_over, t_lift, t_plat, t_next, t_str, t_cmp, t_coach, t_bw, t_data, t_guide = st.tabs(
    ["Overview", "Lift progress", "Plateaus & PRs", "Next workout", "Strength & goals", "Compare",
     "Coach & share", "Bodyweight", "Data & settings", "Guide"])

# ================================================================== OVERVIEW
with t_over:
    c = st.columns(4)
    kpi(c[0], "Workouts logged", f"{len(sess)}", f"about {len(sess) / span_weeks:.1f} per week overall")
    kpi(c[1], "Last 4 weeks", f"{len(l28)}", "workouts · aim for 3-5" if len(l28) < 12 else "workouts, solid")
    kpi(c[2], "Stalled lifts", f"{n_stalled}", "need a change of plan" if n_stalled else "nothing stuck")
    kpi(c[3], "PRs set", f"{len(prs)}", "new best estimated 1RMs")

    # ---- calendar
    section("Training calendar",
            "One square per day. Blue means you trained, and brighter blue means more total weight moved.")
    guide("Each column is a week, each row a weekday. Hover a square for the workout name and volume.",
          "A regular pattern with few empty weeks. Consistency drives results more than any program.",
          "Look for long empty stretches. Those gaps are your biggest lever, bigger than changing exercises.")
    weeks, z, txt = A.calendar_grid(df)
    fig = go.Figure(go.Heatmap(
        z=z, x=weeks, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], text=txt,
        hovertemplate="%{text}<extra></extra>", xgap=3, ygap=3, hoverongaps=True,
        colorscale=[[0, "#184f95"], [1, "#9ec5f4"]], showscale=False))
    style(fig, 310, legend=False)
    fig.update_layout(plot_bgcolor="#242423", margin=dict(l=12, r=16, t=28, b=40))
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category", tickfont=dict(color=INK2, size=11), showticklabels=True)
    fig.update_xaxes(type="category", tickangle=0, nticks=14, side="top")
    show(fig)

    # ---- balance
    section("Are you training every muscle enough?",
            "Hard sets per week for each muscle over your last 4 weeks. A hard set is any working set, "
            "so warm-ups are excluded.")
    guide("Each bar is a muscle. The shaded band is 10-20 sets per week, a commonly used range for building muscle.",
          "Every muscle you care about inside or near the band.",
          "Below the band, add 2-4 sets for that muscle next week. Above it, you may be doing more than you can recover from.")
    bal = A.muscle_balance(df)
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
    fig.update_xaxes(title="Hard sets per week (average of last 4 weeks)", gridcolor=GRID, showgrid=True, range=[0, max(24, bal["sets_per_week"].max() * 1.15)])
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
        st.markdown('<div style="line-height:2.2">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
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
    c = st.columns(4)
    best, cur = e["e1rm"].max(), e["e1rm"].iloc[-1]
    chg = (cur / e["e1rm"].iloc[0] - 1) * 100 if len(e) > 1 else 0
    kpi(c[0], "Best e1RM", f"{best:.0f} {unit}", "your strongest estimated max")
    kpi(c[1], "Latest e1RM", f"{cur:.0f} {unit}", f"{e['day'].iloc[-1]:%d %b %Y}")
    kpi(c[2], "Change since first", f"{chg:+.0f}%", f"over {len(e)} sessions")
    kpi(c[3], "PRs on this lift", f"{int(e['is_pr'].sum())}", "times you beat your best")

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
        fig.update_xaxes(title="Change in estimated 1RM (%)", showgrid=True, gridcolor=GRID,
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
                f'<tr><td>{escape(r["Exercise"])}{" " + chip("crit", "Stalled") if r["Stalled"] else ""}</td>'
                f'<td>{r["Sets"]}</td><td class="t">{escape(r["Target"])}</td><td class="n">{escape(r["Note"])}</td></tr>'
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
    guide(f"Each card shows last session, then the weight and reps to aim for. Your rep range is {low}-{high} (change it in the sidebar).",
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

# ================================================================== BODYWEIGHT
with t_bw:
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

# ================================================================== GUIDE
with t_guide:
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


# ================================================================== DATA & SETTINGS
with t_data:
    section("Your data", "What was loaded, and how to keep your own copy. Repcompass never stores your files on a server.")
    if use_demo:
        st.info("You are looking at demo data.")
    else:
        for info in infos:
            st.markdown(f'<div class="card"><h4>{escape(str(info["name"]))}</h4>'
                        f'<div class="row" style="margin-top:0">{info["source"]} export · {info["rows"]} set rows</div></div>',
                        unsafe_allow_html=True)
        if raw_export is not None:
            label = "Download combined history (CSV)" if len(infos) > 1 else "Download a clean copy (CSV)"
            st.download_button(label, raw_export.to_csv(index=False).encode("utf-8"),
                               file_name="repcompass_history.csv", mime="text/csv")
    st.caption("Tip: export from your app each month, upload the new file together with last month's, and download the "
               "combined history to keep. Workouts that appear in several files are only counted once, and the file listed last wins.")

    section("Save and restore your setup",
            "Your goals, muscle fixes, bodyweight and standards table live in this browser tab only. Download them so you can restore them next visit.")
    setup = {"app": "repcompass", "version": 1, "goals": st.session_state.get("goals", []),
             "muscle_overrides": overrides, "bodyweight": st.session_state.get("bw"), "sex": st.session_state.get("sex")}
    cc = st.columns(2)
    cc[0].download_button("Download my setup (JSON)", json.dumps(setup, indent=2).encode("utf-8"),
                          file_name="repcompass_setup.json", mime="application/json")
    up_setup = cc[1].file_uploader("Restore a setup file", type="json", key="setup_up", label_visibility="collapsed")
    if up_setup is not None and st.session_state.get("_setup_done") != up_setup.file_id:
        try:
            loaded = json.loads(up_setup.getvalue().decode("utf-8"))
            if loaded.get("app") != "repcompass":
                raise ValueError("not a Repcompass setup file")
            st.session_state["goals"] = [g for g in loaded.get("goals", []) if {"exercise", "target"} <= set(g)]
            st.session_state["muscle_overrides"] = {k: v for k, v in loaded.get("muscle_overrides", {}).items()
                                                    if v in F.MUSCLE_CHOICES}
            if isinstance(loaded.get("bodyweight"), (int, float)):
                st.session_state["bw"] = float(loaded["bodyweight"])
            if loaded.get("sex") in ("Male", "Female"):
                st.session_state["sex"] = loaded["sex"]
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

    section("Why there are no accounts", "A quick note on how this app handles your data.")
    st.markdown(
        "- Your file is read in memory to draw the charts and is not saved by the app.\n"
        "- Saved history across visits would need accounts and a database holding your training data. "
        "Instead, download your combined history and setup above and keep them yourself.\n"
        "- Running the app locally keeps everything on your own computer.")


# ================================================================== STRENGTH & GOALS
with t_str:
    section("Strength levels",
            "How your bench, squat, deadlift and overhead press compare with common bodyweight standards.")
    guide("Each card is one lift. The bar shows how far you are from the next level.",
          "Intermediate or better on the lifts you train most.",
          "Chase the lift with the shortest bar first. Beginners often level up fastest on their weakest lift.")
    c1, c2 = st.columns(2)
    bwt = c1.number_input(f"Your bodyweight ({unit})", 30.0, 300.0, 70.0 if unit == "kg" else 155.0, step=0.5, key="bw")
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

        k = st.columns(4)
        kpi(k[0], "Workouts", f"{b_['sessions']}", delta("sessions"))
        kpi(k[1], "Hard sets", f"{b_['hard_sets']}", delta("hard_sets"))
        kpi(k[2], f"Volume ({unit})", f"{b_['volume']:,.0f}", delta("volume"))
        kpi(k[3], "New PRs", f"{b_['prs']}", delta("prs"))
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
            fig.update_xaxes(title="Change in estimated 1RM (%)", showgrid=True, gridcolor=GRID,
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
    bal_c = A.muscle_balance(df)
    review = F.weekly_review(df, plateaus, deload, bal_c, st.session_state.get("goals", []), unit)
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
