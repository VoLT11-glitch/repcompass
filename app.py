"""Lift Brain: turn a Lyfta export into clear answers about your training.

Run:  python -m streamlit run app.py
"""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as A

st.set_page_config(page_title="Lift Brain", page_icon="🏋️", layout="wide")

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
    st.markdown("### 🏋️ Lift Brain")
    st.caption("Your Lyfta data, explained.")
    st.markdown("**1. Your data**")
    up = st.file_uploader("Lyfta CSV export", type="csv", label_visibility="collapsed")
    use_demo = st.toggle("Use demo data", value=False, help="Fake 30-session history so you can explore the app.")
    if up is None and not use_demo and auto:
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

if use_demo:
    df = A.load_demo()
elif up is not None:
    df = A.load(up)
elif auto:
    df = A.load(auto[0])
else:
    df = None


# ------------------------------------------------------------------ onboarding (no data yet)
def onboarding():
    st.markdown(
        '<div class="hero"><div class="kicker">Lift Brain</div><h1>Know if your training is working.</h1>'
        "<p>Upload your Lyfta export and get plain-English answers: are you getting stronger, which lifts "
        "have stalled, are you training every muscle enough, and what weight to try next session.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Get started in 3 steps")
    for n, t, d in [
        ("1", "Export your data from Lyfta", "In the Lyfta app open Settings → Export data and choose CSV. Save the file to your computer."),
        ("2", "Upload it in the sidebar", "Drag the CSV into the box on the left. Nothing leaves your computer when you run the app locally."),
        ("3", "Read the tabs left to right", "Overview tells you the big picture, then dig into each lift, your plateaus, and next session's targets."),
    ]:
        st.markdown(f'<div class="step"><div class="n">{n}</div><div><b>{t}</b><p>{d}</p></div></div>', unsafe_allow_html=True)
    st.info("No export yet? Switch on **Use demo data** in the sidebar to explore with a fake training history.")


if df is None:
    onboarding()
    st.stop()


# ------------------------------------------------------------------ shared numbers
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

st.markdown(
    f'<div class="hero"><div class="kicker">Lift Brain</div><h1>Your training, explained.</h1>'
    f"<p>{len(sess)} workouts logged from {first_day:%d %b %Y} to {last_day:%d %b %Y}. "
    f"Read the tabs left to right. Each one starts with what it shows and ends with what to do.</p></div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Lift progress", "Plateaus & PRs", "Next workout", "Bodyweight", "Guide"])

# ================================================================== OVERVIEW
with tabs[0]:
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
with tabs[1]:
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

# ================================================================== PLATEAUS & PRs
with tabs[2]:
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
    if len(prs):
        t = prs.copy()
        t["Date"] = t["Date"].dt.strftime("%d %b %Y")
        t["Exercise"] = t["Exercise"].str.strip()
        t = t.rename(columns={"e1RM": f"New e1RM ({unit})", "Prev best": f"Old best ({unit})", "Top weight": f"Top weight ({unit})"})
        st.dataframe(t, width="stretch", hide_index=True)
    else:
        st.write("No PRs yet.")

# ================================================================== NEXT WORKOUT
with tabs[3]:
    section("What to lift next session",
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
with tabs[4]:
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
with tabs[5]:
    section("Plain-English guide", "Every term used in this app, and the assumptions behind the numbers.")
    terms = [
        ("Estimated 1RM (e1RM)", "A guess at the heaviest single rep you could do, worked out from a set with more reps. Formula: weight × (1 + reps ÷ 30). It lets you compare sessions even when the reps differ. Sets above 12 reps are ignored."),
        ("Hard set", "Any set that is not a warm-up and has reps logged. Counting hard sets per muscle is a simple way to check you train everything enough."),
        ("Volume", "Weight × reps added up. 3 sets of 10 reps at 50 is 1,500 volume. Used to compare sessions."),
        ("PR (personal record)", "A session where a lift's e1RM beat everything you did before it."),
        ("Plateau / stalled", "The last 3 sessions of a lift did not beat the best before them."),
        ("Double progression", "First fill your rep range at one weight. Once every set hits the top of the range, add weight and start at the bottom again."),
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
