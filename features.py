"""Extra analysis for Repcompass: deloads, strength levels, goals, comparisons, session plans.

No Streamlit imports, so everything here can be tested on its own.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import analysis as A

# ------------------------------------------------------------------ deload / fatigue


def _windows(df: pd.DataFrame, n: int = 8):
    """n rolling 7-day windows ending on the last logged day. Index 0 is the most recent."""
    end = df["day"].max()
    return [(end - pd.Timedelta(days=7 * (k + 1)), end - pd.Timedelta(days=7 * k)) for k in range(n)]


def deload_check(df: pd.DataFrame) -> dict:
    """Is fatigue building up? Looks at the last 8 weeks.

    Signals: (1) how many lifts have a lower best e1RM in the last 2 weeks than in the 4 weeks
    before that, (2) how many hard weeks in a row you have trained without a lighter week.
    """
    span = (df["day"].max() - df["day"].min()).days
    if span < 28:
        return {"status": "Not enough data", "reasons": ["Needs at least 4 weeks of history."],
                "weeks": pd.DataFrame(), "share_down": 0.0, "compared": 0, "down": [], "streak": 0}
    wins = _windows(df, 8)
    h = A.hard_sets(df)
    sets = [int(((h["day"] > a) & (h["day"] <= b)).sum()) for a, b in wins]
    weeks = pd.DataFrame({"window_end": [b for _, b in wins], "sets": sets}).iloc[::-1].reset_index(drop=True)

    w = A.working_sets(df)
    w = w[w["e1rm"].notna()]
    down, compared = [], 0
    for ex, g in w.groupby("Exercise"):
        recent = g[(g["day"] > wins[1][0]) & (g["day"] <= wins[0][1])]["e1rm"]
        base = g[(g["day"] > wins[5][0]) & (g["day"] <= wins[2][1])]["e1rm"]
        if recent.empty or base.empty:
            continue
        compared += 1
        if recent.max() < base.max() * 0.97:
            down.append(ex.strip())
    share = len(down) / compared if compared else 0.0

    nonzero = [s for s in sets if s > 0]
    med = float(np.median(nonzero)) if nonzero else 0.0
    streak = 0
    for s in sets:  # most recent first
        if med and s >= 0.75 * med:
            streak += 1
        else:
            break

    reasons = []
    if compared >= 3:
        reasons.append(f"{len(down)} of {compared} lifts have a lower best e1RM over the last 2 weeks than in the 4 weeks before.")
    else:
        reasons.append("Too few repeated lifts to compare strength trends.")
    reasons.append(f"You have trained at a normal or higher workload for {streak} week(s) in a row.")
    if compared >= 3 and share >= 0.5:
        status = "Deload suggested"
    elif streak >= 6 and share >= 0.34:
        status = "Deload suggested"
    elif (compared >= 3 and share >= 0.34) or streak >= 5:
        status = "Watch"
    else:
        status = "Fine"
    return {"status": status, "reasons": reasons, "weeks": weeks, "share_down": share,
            "compared": compared, "down": down, "streak": streak}


# ------------------------------------------------------------------ strength standards
LEVELS = ["Beginner", "Novice", "Intermediate", "Advanced", "Elite"]
# Rough bodyweight multiples for a one-rep max. Guidelines only, never a verdict.
STANDARDS = {
    "Male": {
        "Bench press": [0.50, 0.75, 1.25, 1.75, 2.00],
        "Squat": [0.75, 1.25, 1.75, 2.25, 2.75],
        "Deadlift": [1.00, 1.50, 2.00, 2.50, 3.00],
        "Overhead press": [0.35, 0.55, 0.80, 1.05, 1.30],
    },
    "Female": {
        "Bench press": [0.25, 0.50, 0.75, 1.00, 1.50],
        "Squat": [0.50, 0.75, 1.25, 1.50, 2.00],
        "Deadlift": [0.50, 1.00, 1.25, 1.75, 2.25],
        "Overhead press": [0.20, 0.35, 0.50, 0.70, 0.90],
    },
}
LIFT_MATCH = {
    "Bench press": (["bench press"], ["dumbbell", "incline", "decline", "close", "smith", "machine", "cable"]),
    "Squat": (["squat"], ["hack", "split", "goblet", "bulgarian", "smith", "machine", "jump", "overhead"]),
    "Deadlift": (["deadlift"], ["romanian", "stiff", "straight", "dumbbell", "single"]),
    "Overhead press": (["overhead press", "military press", "shoulder press"],
                       ["dumbbell", "machine", "arnold", "smith"]),
}


def strength_levels(df: pd.DataFrame, bodyweight: float, sex: str = "Male", days: int = 120) -> pd.DataFrame:
    """Level for the big four lifts, from best e1RM in the last `days` days (falls back to all time)."""
    w = A.working_sets(df)
    w = w[w["e1rm"].notna()]
    cutoff = df["day"].max() - pd.Timedelta(days=days)
    rows = []
    for lift, (any_k, none_k) in LIFT_MATCH.items():
        names = [e for e in w["Exercise"].unique()
                 if any(k in e.lower() for k in any_k) and not any(k in e.lower() for k in none_k)]
        if not names:
            continue
        sub = w[w["Exercise"].isin(names)]
        recent = sub[sub["day"] >= cutoff]
        pool = recent if len(recent) else sub
        best = pool.loc[pool["e1rm"].idxmax()]
        ratio = best["e1rm"] / bodyweight
        t = STANDARDS[sex][lift]
        idx = sum(ratio >= x for x in t)  # 0 = untrained
        level = "Untrained" if idx == 0 else LEVELS[idx - 1]
        if idx < len(t):
            nxt, need = LEVELS[idx], t[idx] * bodyweight
            lo = t[idx - 1] if idx > 0 else 0.0
            prog = float(np.clip((ratio - lo) / (t[idx] - lo), 0, 1))
        else:
            nxt, need, prog = "Top level", np.nan, 1.0
        rows.append({"Lift": lift, "Exercise": best["Exercise"].strip(), "Best e1RM": best["e1rm"],
                     "x Bodyweight": ratio, "Level": level, "Next level": nxt,
                     "Needed e1RM": need, "Progress": prog})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ volume per lift
def lift_volume(df: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    """Hard sets per week for each exercise over the last `days`, plus who you have dropped."""
    end = df["day"].max()
    h = A.hard_sets(df)
    rows = []
    for ex, g in h.groupby("Exercise"):
        recent = g[g["day"] > end - pd.Timedelta(days=days)]
        total_sessions = g["day"].nunique()
        last = g["day"].max()
        since = (end - last).days
        if since > 21 and total_sessions >= 3:
            status = "Dropped"
        elif len(recent) and recent["day"].nunique() == 1 and total_sessions >= 3:
            status = "Rare"
        elif len(recent):
            status = "Active"
        else:
            status = "Inactive"
        rows.append({"Exercise": ex.strip(), "Muscle": g["muscle"].iloc[0],
                     "Sets / week": len(recent) * 7.0 / days,
                     "Sessions (period)": recent["day"].nunique(),
                     "Last done (days ago)": since, "Total sessions": total_sessions, "Status": status})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"Dropped": 0, "Rare": 1, "Active": 2, "Inactive": 3}
    out["_o"] = out["Status"].map(order)
    return out.sort_values(["_o", "Sets / week"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def lift_weekly_matrix(df: pd.DataFrame, weeks: int = 8, top: int = 12) -> pd.DataFrame:
    """Hard sets per exercise in each of the last `weeks` rolling weeks (oldest first)."""
    wins = _windows(df, weeks)[::-1]
    h = A.hard_sets(df)
    rows = {}
    for ex, g in h.groupby("Exercise"):
        rows[ex.strip()] = [int(((g["day"] > a) & (g["day"] <= b)).sum()) for a, b in wins]
    m = pd.DataFrame(rows, index=[f"{b:%d %b}" for _, b in wins]).T
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index[:top]]
    return m


# ------------------------------------------------------------------ comparing two periods
def period_stats(df: pd.DataFrame, start, end) -> dict:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    d = df[(df["day"] >= start) & (df["day"] <= end)]
    if d.empty:
        return {"sessions": 0, "hard_sets": 0, "volume": 0.0, "prs": 0, "days": 0,
                "per_week": 0.0, "muscle_sets": {}, "best": pd.Series(dtype=float)}
    w = A.working_sets(d)
    h = A.hard_sets(d)
    prs = A.pr_timeline(df)
    n_pr = int(((prs["Date"] >= start) & (prs["Date"] <= end)).sum()) if len(prs) else 0
    weeks = max(((end - start).days + 1) / 7.0, 1e-9)
    best = w[w["e1rm"].notna()].groupby("Exercise")["e1rm"].max()
    return {"sessions": int(d["session"].nunique()), "hard_sets": len(h), "volume": float(w["volume"].sum()),
            "prs": n_pr, "days": int(d["day"].nunique()), "per_week": d["session"].nunique() / weeks,
            "muscle_sets": h.groupby("muscle").size().to_dict(), "best": best}


def compare_periods(df: pd.DataFrame, a: tuple, b: tuple) -> dict:
    """Compare period A against period B. Each is (start, end)."""
    sa, sb = period_stats(df, *a), period_stats(df, *b)
    lifts = pd.DataFrame({"A": sa["best"], "B": sb["best"]}).dropna()
    if len(lifts):
        lifts["Change %"] = (lifts["B"] / lifts["A"] - 1) * 100
        lifts = lifts.sort_values("Change %", ascending=False).reset_index().rename(columns={"Exercise": "Exercise"})
        lifts["Exercise"] = lifts["Exercise"].str.strip()
    muscles = pd.DataFrame({"A": pd.Series(sa["muscle_sets"]), "B": pd.Series(sb["muscle_sets"])}).fillna(0)
    muscles = muscles.reindex([m for m in A.MUSCLE_ORDER if m in muscles.index])
    return {"a": sa, "b": sb, "lifts": lifts, "muscles": muscles}


# ------------------------------------------------------------------ rep records + PR feed
def rep_records(df: pd.DataFrame, exercise: str, max_reps: int = 15) -> pd.DataFrame:
    """Heaviest weight you have lifted for each rep count, and when."""
    w = A.working_sets(df)
    w = w[w["Exercise"] == exercise].copy()
    w["Reps"] = w["Reps"].astype(int)
    w = w[w["Reps"] <= max_reps]
    rows = []
    for r, g in w.groupby("Reps"):
        top = g.loc[g["Weight"].idxmax()]
        rows.append({"Reps": int(r), "Best weight": float(top["Weight"]), "Date": top["day"]})
    return pd.DataFrame(rows).sort_values("Reps").reset_index(drop=True) if rows else pd.DataFrame()


def pr_feed(df: pd.DataFrame) -> pd.DataFrame:
    p = A.pr_timeline(df)
    if p.empty:
        return p
    p = p.copy()
    p["Gain"] = p["e1RM"] - p["Prev best"]
    p["Gain %"] = p["Gain"] / p["Prev best"] * 100
    return p


# ------------------------------------------------------------------ goals
def goal_projection(df: pd.DataFrame, exercise: str, target: float, target_date=None, window_days: int = 120) -> dict:
    """Project when a lift reaches a target e1RM, from the trend of the last `window_days`."""
    e = A.e1rm_by_session(df, exercise).sort_values("day")
    if e.empty:
        return {"ok": False, "note": "No usable sets for this lift."}
    current = float(e["e1rm"].tail(3).max())
    last = e["day"].iloc[-1]
    out = {"ok": True, "current": current, "target": float(target),
           "progress": float(np.clip(current / target, 0, 1)), "reached": current >= target,
           "eta": None, "slope_week": None, "on_track": None, "note": ""}
    if out["reached"]:
        out["note"] = "Goal reached. Time to set a bigger one."
        return out
    r = e[e["day"] >= last - pd.Timedelta(days=window_days)]
    if len(r) < 3:
        out["note"] = "Needs at least 3 sessions in the last 4 months to project a date."
        return out
    x = (r["day"] - r["day"].iloc[0]).dt.days.to_numpy(dtype=float)
    slope = float(np.polyfit(x, r["e1rm"].to_numpy(), 1)[0])  # per day
    out["slope_week"] = slope * 7
    if slope <= 0:
        out["note"] = "No upward trend in the last 4 months, so no date can be projected."
        return out
    days_needed = (target - current) / slope
    if days_needed > 365 * 5:
        out["note"] = "At the current pace this is more than 5 years away."
        return out
    out["eta"] = (last + pd.Timedelta(days=days_needed)).normalize()
    if target_date is not None:
        td = pd.Timestamp(target_date)
        out["on_track"] = bool(out["eta"] <= td)
        weeks_left = max((td - last).days / 7.0, 0.1)
        out["needed_week"] = (target - current) / weeks_left
    return out


# ------------------------------------------------------------------ full-session plan
def plan_next_session(df: pd.DataFrame, low: int = 8, high: int = 12, step: float = 2.5,
                      plateaus: pd.DataFrame | None = None, title: str | None = None) -> dict:
    """Guess your next workout from your usual rotation, then set a target for each lift."""
    sess = df.drop_duplicates("session").sort_values("Date")
    titles = sess["Title"].tolist()
    if not titles:
        return {}
    last_title = titles[-1]
    trans = Counter(zip(titles[:-1], titles[1:]))
    cand = {b: c for (a, b), c in trans.items() if a == last_title}
    if title is not None and title in titles:
        next_title, how = title, "You picked this workout."
    elif cand:
        next_title = max(cand, key=cand.get)
        how = f"After a {last_title} day you most often train {next_title}."
    else:
        counts = Counter(titles)
        repeated = [t for t, c in counts.items() if c >= 2] or list(counts)
        lastdone = {t: sess[sess["Title"] == t]["Date"].max() for t in repeated}
        next_title = min(lastdone, key=lastdone.get)
        how = f"{next_title} is the workout you have not done for the longest."
    ref = sess[sess["Title"] == next_title].iloc[-1]
    rows_df = df[df["session"] == ref["session"]]
    stalled = set()
    if plateaus is not None and len(plateaus):
        stalled = {e.strip() for e in plateaus.loc[plateaus["Status"] == "Stalled", "Exercise"]}

    rows = []
    for ex in rows_df["Exercise"].drop_duplicates():
        g = rows_df[(rows_df["Exercise"] == ex) & ~rows_df["is_warmup"]]
        if g.empty or g["is_cardio"].iloc[0]:
            continue
        n_sets = len(g)
        name = ex.strip()
        if g["is_timed"].iloc[0]:
            secs = g["seconds"].max()
            if pd.notna(secs):
                rows.append({"Exercise": name, "Sets": n_sets, "Target": f"{int(secs) + 5} s hold",
                             "Note": "Add 5 seconds.", "Stalled": False})
            continue
        s = A.suggest_next(df, ex, low, high, step)
        if s:
            note = s["Why"]
            if name in stalled:
                note += " Stalled lift: try a back-off set or a few weeks in a different rep range."
            rows.append({"Exercise": name, "Sets": n_sets,
                         "Target": f"{s['Next weight (kg)']:g} x {s['Target reps per set']}",
                         "Note": note, "Stalled": name in stalled})
        elif g["is_bodyweight"].iloc[0]:
            best = int(g["Reps"].max())
            rows.append({"Exercise": name, "Sets": n_sets, "Target": f"{best + 1} reps (best set)",
                         "Note": "Bodyweight move: aim for one more rep on your best set.", "Stalled": name in stalled})
    return {"title": next_title, "based_on": ref["Date"], "how": how, "table": pd.DataFrame(rows)}


# ------------------------------------------------------------------ weekly review + AI prompt
def weekly_review(df: pd.DataFrame, plateaus: pd.DataFrame, deload: dict, bal: pd.DataFrame,
                  goals: list[dict] | None = None, unit: str = "kg") -> dict:
    """Plain-English review from computed stats. Returns bullets [(kind, text)] and a copyable AI prompt."""
    end = df["day"].max()
    sess = df.drop_duplicates("session")
    this_w = sess[sess["day"] > end - pd.Timedelta(days=7)]
    span_weeks = max((end - df["day"].min()).days / 7.0, 1.0)
    avg = len(sess) / span_weeks
    bullets = []
    n = len(this_w)
    kind = "good" if n >= round(avg) and n > 0 else "warn"
    bullets.append((kind, f"You trained {n} time(s) in the last 7 days. Your overall average is {avg:.1f} per week."))

    prs = A.pr_timeline(df)
    recent = prs[prs["Date"] > end - pd.Timedelta(days=14)] if len(prs) else prs
    if len(recent):
        names = ", ".join(sorted({e.strip() for e in recent["Exercise"]})[:4])
        bullets.append(("good", f"{len(recent)} new PR(s) in the last 2 weeks: {names}."))
    else:
        bullets.append(("info", "No new PRs in the last 2 weeks."))

    if len(plateaus):
        st_ = plateaus[plateaus["Status"] == "Stalled"]
        if len(st_):
            names = ", ".join(e.strip() for e in st_["Exercise"].head(3))
            bullets.append(("crit", f"{len(st_)} stalled lift(s): {names}. Change one thing at a time and give it 3 sessions."))
        else:
            bullets.append(("good", "No stalled lifts right now."))

    if len(bal):
        low_m = [r["muscle"] for _, r in bal.iterrows() if r["status"] == "Low" and r["muscle"] != "Calves"]
        high_m = [r["muscle"] for _, r in bal.iterrows() if r["status"] == "High"]
        if low_m:
            bullets.append(("warn", f"Under-trained (below 10 sets/week): {', '.join(low_m)}. Add 2-4 sets for each."))
        if high_m:
            bullets.append(("info", f"Above 20 sets/week: {', '.join(high_m)}. Make sure you recover from it."))
        if not low_m and not high_m:
            bullets.append(("good", "Muscle volume is inside the 10-20 sets/week band."))

    kind = {"Deload suggested": "crit", "Watch": "warn"}.get(deload["status"], "good")
    txt = {"Deload suggested": "Fatigue signals say a lighter week would help.",
           "Watch": "Early fatigue signs. Keep an eye on the next 1-2 weeks.",
           "Fine": "No fatigue warning. Keep going."}.get(deload["status"], "Not enough history for a fatigue check yet.")
    bullets.append((kind, txt))

    for g in goals or []:
        p = goal_projection(df, g["exercise"], g["target"], g.get("date"))
        if not p.get("ok"):
            continue
        name = g["exercise"].strip()
        if p["reached"]:
            bullets.append(("good", f"Goal reached: {name} e1RM {g['target']:g} {unit}."))
        elif p["eta"] is not None:
            bullets.append(("info", f"{name}: {p['current']:.0f} of {g['target']:g} {unit} e1RM. At this pace: {p['eta']:%d %b %Y}."))
        else:
            bullets.append(("warn", f"{name}: {p['current']:.0f} of {g['target']:g} {unit} e1RM. {p['note']}"))

    lines = "\n".join(f"- {t}" for _, t in bullets)
    prompt = (
        "You are an evidence-based strength coach. Below is a summary of my last training weeks, "
        f"computed from my workout log (weights in {unit}, estimated 1RM uses the Epley formula, "
        "no RPE data). Give me a short weekly review: what went well, the 3 most important things to "
        "change next week, and one sentence on whether to deload.\n\n"
        f"Period: {df['day'].min():%d %b %Y} to {end:%d %b %Y}, {len(sess)} workouts.\n{lines}"
    )
    return {"bullets": bullets, "prompt": prompt}


# ------------------------------------------------------------------ muscle overrides
MUSCLE_CHOICES = A.MUSCLE_ORDER + ["Cardio", "Other"]


def apply_muscle_overrides(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    if not mapping:
        return df
    out = df.copy()
    out["muscle"] = out["Exercise"].map(mapping).fillna(out["muscle"])
    return out


def exercise_muscles(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Exercise").agg(Muscle=("muscle", "first"), Sets=("Exercise", "size")).reset_index()
    return g.sort_values("Sets", ascending=False).reset_index(drop=True)
