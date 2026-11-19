"""Shareable summary card (PNG) and monthly report (PDF), drawn with matplotlib.

No Streamlit imports. Both functions return bytes ready for st.download_button.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import analysis as A  # noqa: E402
import features as F  # noqa: E402

BG, CARD, GRID = "#121211", "#1c1c1b", "#2c2c2a"
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#8d8c85"
BLUE, GOLD = A.SERIES[0], "#fab219"


def months_available(df: pd.DataFrame) -> list[pd.Period]:
    return sorted(df["day"].dt.to_period("M").unique(), reverse=True)


def _month_bounds(month: pd.Period):
    return month.start_time.normalize(), month.end_time.normalize()


def _compact(v: float, unit: str) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M {unit}"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k {unit}"
    return f"{v:.0f} {unit}"


def _week_streak(df: pd.DataFrame) -> int:
    """Consecutive calendar weeks (ending with the latest week) with at least one workout."""
    weeks = set(df["week"].unique())
    cur = df["week"].max()
    n = 0
    while cur in weeks:
        n += 1
        cur -= pd.Timedelta(days=7)
    return n


def month_summary(df: pd.DataFrame, month: pd.Period) -> dict:
    start, end = _month_bounds(month)
    st = F.period_stats(df, start, end)
    prev_start, prev_end = _month_bounds(month - 1)
    pv = F.period_stats(df, prev_start, prev_end)
    prs = F.pr_feed(df)
    mprs = prs[(prs["Date"] >= start) & (prs["Date"] <= end)] if len(prs) else prs
    w = A.working_sets(df)
    w = w[w["e1rm"].notna()]
    best_gain = None
    for ex, g in w.groupby("Exercise"):
        before = g[g["day"] < start]["e1rm"]
        inside = g[(g["day"] >= start) & (g["day"] <= end)]["e1rm"]
        if len(before) and len(inside):
            pct = (inside.max() / before.max() - 1) * 100
            if best_gain is None or pct > best_gain[1]:
                best_gain = (ex.strip(), pct, inside.max())
    return {"start": start, "end": end, "stats": st, "prev": pv, "prs": mprs, "best_gain": best_gain,
            "streak": _week_streak(df[df["day"] <= end])}


def share_card(df: pd.DataFrame, month: pd.Period, unit: str = "kg", fmt: str = "post") -> bytes:
    """1080x1350 post or 1080x1920 story. Dark card with the month's headline numbers."""
    W, H = 1080, 1350 if fmt == "post" else 1920
    off = (H - 1350) // 2
    ms = month_summary(df, month)
    st = ms["stats"]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def text(x, y, s, size, color=INK, weight="normal", ha="left", va="top"):
        ax.text(x, y + off, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)

    def rect(x, y, w, h, color=CARD, r=28):
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (x, y + off), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=color, ec=GRID, lw=2))

    text(80, 90, "REPCOMPASS", 22, BLUE, "bold")
    text(80, 140, month.strftime("%B %Y"), 64, INK, "bold")
    text(80, 252, "My training month", 26, MUTED)

    tiles = [("Workouts", f"{st['sessions']}", f"{ms['prev']['sessions']} last month"),
             ("Hard sets", f"{st['hard_sets']}", f"{ms['prev']['hard_sets']} last month"),
             ("New PRs", f"{st['prs']}", "estimated 1RM bests"),
             ("Volume", _compact(st["volume"], unit), "weight x reps")]
    for i, (lab, val, sub) in enumerate(tiles):
        x, y = 80 + (i % 2) * 470, 330 + (i // 2) * 250
        rect(x, y, 450, 220)
        text(x + 36, y + 30, lab.upper(), 18, MUTED, "bold")
        text(x + 36, y + 70, val, 58, INK, "bold")
        text(x + 36, y + 160, sub, 18, MUTED)

    y0 = 850
    rect(80, y0, 920, 200)
    if ms["best_gain"]:
        name, pct, e1 = ms["best_gain"]
        text(116, y0 + 28, "BIGGEST GAIN", 18, MUTED, "bold")
        text(116, y0 + 66, name[:26], 36, INK, "bold")
        text(116, y0 + 125, f"{pct:+.1f}% on estimated 1RM  ·  now {e1:.0f} {unit}", 24, GOLD, "bold")
    else:
        text(116, y0 + 28, "BIGGEST GAIN", 18, MUTED, "bold")
        text(116, y0 + 80, "Not enough repeated lifts yet", 28, INK2)

    # weekly sessions mini chart
    wk = df[(df["day"] >= ms["start"]) & (df["day"] <= ms["end"])].drop_duplicates("session").groupby("week").size()
    if len(wk):
        text(80, 1090, "WORKOUTS PER WEEK", 18, MUTED, "bold")
        cx = fig.add_axes([80 / W, (H - (1140 + off) - 120) / H, 920 / W, 120 / H], facecolor="none")
        cx.bar(range(len(wk)), wk.values, color=BLUE, width=0.55)
        for i, v in enumerate(wk.values):
            cx.text(i, v + 0.05, str(v), ha="center", va="bottom", color=INK2, fontsize=14)
        cx.set_ylim(0, max(wk.values.max() * 1.35, 1))
        cx.axis("off")

    text(80, 1280, f"{ms['streak']} week streak" if ms["streak"] > 1 else "Keep showing up", 24, INK2, "bold")
    text(1000, 1280, "repcompass.streamlit.app", 20, MUTED, ha="right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- PDF report
def _page(title: str, sub: str):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.07, 0.955, title, fontsize=22, fontweight="bold", color="#111")
    fig.text(0.07, 0.93, sub, fontsize=10.5, color="#666")
    return fig


def _table(fig, x, y, rows: list[list[str]], widths: list[float], header: list[str], size=9, step=0.022):
    cx = x
    for h, w in zip(header, widths):
        fig.text(cx, y, h, fontsize=size, fontweight="bold", color="#555")
        cx += w
    fig.add_artist(plt.Line2D([x, x + sum(widths)], [y - 0.006, y - 0.006], color="#bbb", lw=0.8))
    for i, r in enumerate(rows):
        cx = x
        for c, w in zip(r, widths):
            fig.text(cx, y - step * (i + 1), str(c), fontsize=size, color="#111")
            cx += w


def monthly_report_pdf(df: pd.DataFrame, month: pd.Period, unit: str = "kg", low: int = 8, high: int = 12,
                       step: float = 2.5) -> bytes:
    ms = month_summary(df, month)
    st, pv = ms["stats"], ms["prev"]
    plateaus = A.plateau_report(df)
    deload = F.deload_check(df)
    plan = F.plan_next_session(df, low, high, step, plateaus)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---- page 1: numbers
        fig = _page(f"Training report: {month.strftime('%B %Y')}", f"Made with Repcompass · {ms['start']:%d %b} to {ms['end']:%d %b %Y}")
        kp = [("Workouts", st["sessions"], pv["sessions"]), ("Hard sets", st["hard_sets"], pv["hard_sets"]),
              ("New PRs", st["prs"], pv["prs"]), (f"Volume ({unit})", f"{st['volume']:,.0f}", f"{pv['volume']:,.0f}")]
        for i, (lab, v, p) in enumerate(kp):
            x = 0.07 + i * 0.225
            fig.add_artist(plt.matplotlib.patches.FancyBboxPatch((x, 0.82), 0.2, 0.075, boxstyle="round,pad=0,rounding_size=0.01",
                                                                 transform=fig.transFigure, fc="#f4f4f2", ec="#ddd"))
            fig.text(x + 0.012, 0.875, lab.upper(), fontsize=7.5, color="#777", fontweight="bold")
            fig.text(x + 0.012, 0.842, str(v), fontsize=17, fontweight="bold", color="#111")
            fig.text(x + 0.012, 0.826, f"last month: {p}", fontsize=7.5, color="#888")

        ax = fig.add_axes([0.2, 0.55, 0.72, 0.22])
        ms_sets = pd.Series(st["muscle_sets"]).reindex(A.MUSCLE_ORDER).dropna()
        ax.barh(ms_sets.index, ms_sets.values, color=[A.MUSCLE_COLORS[m] for m in ms_sets.index], height=0.6)
        ax.invert_yaxis()
        ax.set_title("Hard sets per muscle this month", loc="left", fontsize=11, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for i, v in enumerate(ms_sets.values):
            ax.text(v + 0.5, i, f"{int(v)}", va="center", fontsize=8)

        fig.text(0.07, 0.49, "New PRs this month", fontsize=12, fontweight="bold")
        prs = ms["prs"]
        if len(prs):
            rows = [[r["Exercise"].strip()[:30], f"{r['Date']:%d %b}", f"{r['e1RM']:.0f}", f"+{r['Gain']:.1f}"]
                    for _, r in prs.sort_values("Gain %", ascending=False).head(10).iterrows()]
            _table(fig, 0.07, 0.465, rows, [0.4, 0.15, 0.15, 0.15], ["Lift", "Date", f"e1RM ({unit})", "Gain"])
        else:
            fig.text(0.07, 0.465, "No PRs this month.", fontsize=10, color="#666")

        fig.text(0.07, 0.2, "Fatigue check", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.175, f"Status: {deload['status']}", fontsize=10)
        for i, r in enumerate(deload["reasons"]):
            fig.text(0.07, 0.152 - i * 0.02, "- " + r, fontsize=8.5, color="#555")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: lifts + plan
        fig = _page("Lifts and next session", f"{month.strftime('%B %Y')} · Repcompass")
        fig.text(0.07, 0.89, "Stalled lifts", fontsize=12, fontweight="bold")
        stalled = plateaus[plateaus["Status"] == "Stalled"] if len(plateaus) else plateaus
        if len(stalled):
            rows = [[r["Exercise"].strip()[:32], f"{r['Best e1RM']:.0f}", f"{r['Recent best']:.0f}", r["Days since last PR"]]
                    for _, r in stalled.head(8).iterrows()]
            _table(fig, 0.07, 0.865, rows, [0.4, 0.15, 0.15, 0.2], ["Lift", "Best e1RM", "Recent best", "Days since PR"])
        else:
            fig.text(0.07, 0.865, "None. Nothing is stuck right now.", fontsize=10, color="#666")

        fig.text(0.07, 0.62, "Best lifts: change in estimated 1RM this month vs before", fontsize=12, fontweight="bold")
        c = F.compare_periods(df, (df["day"].min(), ms["start"] - pd.Timedelta(days=1)), (ms["start"], ms["end"]))
        if len(c["lifts"]):
            rows = [[r["Exercise"][:32], f"{r['A']:.0f}", f"{r['B']:.0f}", f"{r['Change %']:+.1f}%"]
                    for _, r in c["lifts"].head(8).iterrows()]
            _table(fig, 0.07, 0.595, rows, [0.4, 0.15, 0.15, 0.15], ["Lift", "Before", "This month", "Change"])
        else:
            fig.text(0.07, 0.595, "Not enough repeated lifts to compare.", fontsize=10, color="#666")

        fig.text(0.07, 0.34, f"Next workout: {plan.get('title', '-')}", fontsize=12, fontweight="bold")
        if plan and len(plan["table"]):
            rows = [[r["Exercise"][:30], r["Sets"], r["Target"]] for _, r in plan["table"].head(8).iterrows()]
            _table(fig, 0.07, 0.315, rows, [0.45, 0.1, 0.3], ["Exercise", "Sets", f"Target ({unit} x reps)"])
        fig.text(0.07, 0.06, "Estimated 1RM uses the Epley formula. Targets use double progression. Guidelines, not rules.",
                 fontsize=7.5, color="#888")
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()
