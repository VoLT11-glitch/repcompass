"""Core analysis for Lyfta CSV exports. No Streamlit imports so it can be tested alone."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Keyword -> muscle group. First match wins, so order matters.
MUSCLE_RULES = [
    ("Calves", ["calf"]),
    ("Core", ["crunch", "plank", "leg hip raise", "leg raise", "ab "]),
    ("Triceps", ["tricep", "skull", "dip", "pushdown"]),
    ("Biceps", ["curl", "preacher"]),
    ("Shoulders", ["lateral raise", "shoulder press", "military", "rear delt", "reverse fly", "face pull"]),
    ("Back", ["pulldown", "pull-up", "pull up", "row", "deadlift"]),
    ("Legs", ["squat", "leg curl", "leg extension", "leg press", "leg wide press", "hip thrust", "lunge", "sled"]),
    ("Chest", ["bench", "fly", "chest", "press"]),
    ("Cardio", ["treadmill", "walking", "run", "cycle"]),
]
# Romanian / straight-leg deadlifts are hamstrings, not back
LEG_OVERRIDES = ["romanian", "straight leg deadlift", "stiff", "leg curl", "leg extension"]

BODYWEIGHT_LIFTS = ["pull-up", "pull up", "elbow dips", "chin-up"]
TIMED_LIFTS = ["plank"]
CARDIO_LIFTS = ["treadmill", "walking"]


def muscle_group(name: str) -> str:
    n = name.lower()
    if any(k in n for k in LEG_OVERRIDES):
        return "Legs"
    for group, keys in MUSCLE_RULES:
        if any(k in n for k in keys):
            return group
    return "Other"


def _parse_time(s) -> float:
    """'1:30' / '10:00' -> seconds. Lyfta stores mm:ss."""
    if not isinstance(s, str) or ":" not in s:
        return np.nan
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return np.nan


def load(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer, na_values=["null", ""])
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = df["Date"].dt.normalize()
    df["session"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M") + " | " + df["Title"].fillna("")
    df["seconds"] = df["Time"].map(_parse_time)
    df["muscle"] = df["Exercise"].map(muscle_group)
    low = df["Exercise"].str.lower()
    df["is_bodyweight"] = low.apply(lambda n: any(k in n for k in BODYWEIGHT_LIFTS))
    df["is_timed"] = low.apply(lambda n: any(k in n for k in TIMED_LIFTS))
    df["is_cardio"] = low.apply(lambda n: any(k in n for k in CARDIO_LIFTS))
    df["is_warmup"] = df["Set Type"].eq("WARMUP_SET")
    df["week"] = df["day"] - pd.to_timedelta(df["day"].dt.weekday, unit="D")
    return df


def epley(weight, reps):
    return weight * (1 + reps / 30.0)


def working_sets(df: pd.DataFrame) -> pd.DataFrame:
    """Loaded, non-warmup sets with real weight and reps. Used for strength maths."""
    m = (
        ~df["is_warmup"]
        & ~df["is_cardio"]
        & ~df["is_timed"]
        & df["Weight"].notna()
        & (df["Weight"] > 0)
        & df["Reps"].notna()
        & (df["Reps"] > 0)
    )
    w = df[m].copy()
    w["e1rm"] = epley(w["Weight"], w["Reps"])
    # Epley overestimates for high reps; keep trend honest
    w.loc[w["Reps"] > 12, "e1rm"] = np.nan
    w["volume"] = w["Weight"] * w["Reps"]
    return w


def hard_sets(df: pd.DataFrame) -> pd.DataFrame:
    """Every non-warmup set that counts as training volume (incl. bodyweight)."""
    m = ~df["is_warmup"] & ~df["is_cardio"] & ~df["is_timed"] & df["Reps"].notna() & (df["Reps"] > 0)
    return df[m].copy()


def weekly_sets_by_muscle(df: pd.DataFrame) -> pd.DataFrame:
    h = hard_sets(df)
    t = h.groupby(["week", "muscle"]).size().unstack(fill_value=0)
    full = pd.date_range(t.index.min(), t.index.max(), freq="7D")
    return t.reindex(full, fill_value=0)


def e1rm_by_session(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    w = working_sets(df)
    w = w[(w["Exercise"] == exercise) & w["e1rm"].notna()]
    return (
        w.groupby("day")
        .agg(e1rm=("e1rm", "max"), top_weight=("Weight", "max"), volume=("volume", "sum"))
        .reset_index()
    )


def pr_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """A row each time an exercise's best e1RM beats everything before it.
    The first session of an exercise is a baseline, not a PR."""
    w = working_sets(df)
    w = w[w["e1rm"].notna()]
    best = w.groupby(["Exercise", "day"]).agg(e1rm=("e1rm", "max"), weight=("Weight", "max")).reset_index()
    rows = []
    for ex, g in best.groupby("Exercise"):
        g = g.sort_values("day")
        run = -np.inf
        first = True
        for _, r in g.iterrows():
            if r["e1rm"] > run + 1e-9:
                if not first:
                    rows.append({"Date": r["day"], "Exercise": ex, "e1RM": round(r["e1rm"], 1),
                                 "Prev best": round(run, 1), "Top weight": r["weight"]})
                run = r["e1rm"]
            first = False
    out = pd.DataFrame(rows)
    return out.sort_values("Date", ascending=False) if len(out) else out


def plateau_report(df: pd.DataFrame, min_sessions: int = 4, window: int = 3) -> pd.DataFrame:
    """Flag lifts where the last `window` sessions did not beat the best before them."""
    w = working_sets(df)
    w = w[w["e1rm"].notna()]
    rows = []
    for ex, g in w.groupby("Exercise"):
        s = g.groupby("day")["e1rm"].max().sort_index()
        if len(s) < min_sessions:
            continue
        before, recent = s.iloc[:-window], s.iloc[-window:]
        if before.empty:
            continue
        best_before, best_recent = before.max(), recent.max()
        prev_max = s.cummax().shift(fill_value=-np.inf)
        last_pr_day = s[s > prev_max + 1e-9].index[-1]
        days_since = (s.index[-1] - last_pr_day).days
        stalled = best_recent <= best_before + 1e-9
        rows.append({
            "Exercise": ex, "Sessions": len(s),
            "Best e1RM": round(s.max(), 1),
            "Recent best": round(best_recent, 1),
            "Days since last PR": days_since,
            "Status": "Stalled" if stalled else ("Progressing" if days_since <= 60 else "Needs more data"),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Status", "Days since last PR"], ascending=[False, False])
    return out


def suggest_next(df: pd.DataFrame, exercise: str, low: int = 8, high: int = 12, step: float = 2.5) -> dict:
    """Double progression using reps only (the export has no RPE).

    Take the last session's working sets at the weight used for the most sets
    (ties go to the heavier weight, so a lone heavy single is ignored).
    - every such set hit `high` reps  -> add `step`, aim for `low` reps
    - otherwise                        -> same weight, aim +1 rep on the weakest set
    - weakest set under `low` reps     -> same weight, build back up (no jump)
    """
    w = working_sets(df)
    w = w[w["Exercise"] == exercise]
    if w.empty:
        return {}
    last_day = w["day"].max()
    last = w[w["day"] == last_day]
    counts = last.groupby("Weight").size()
    # the working weight is the one used for the most sets; a lone heavy single doesn't count
    top_w = counts[counts == counts.max()].index.max()
    top = last[last["Weight"] == top_w]
    reps = top["Reps"].astype(int).tolist()
    mn, mx = min(reps), max(reps)
    if mx <= 3:
        return {
            "Last session": last_day.date(),
            "Last working sets": f"{top_w:g} kg x {', '.join(map(str, reps))}",
            "Next weight (kg)": float(top_w),
            "Target reps per set": int(mx),
            "Why": "Heavy low-rep day, so the rep-range rule doesn't apply. "
                   "Repeat it, or add a small jump if it moved fast.",
        }
    if mn >= high:
        nxt_w, target, why = top_w + step, low, f"All sets hit {high}+ reps, so add weight."
    elif mn < low:
        nxt_w, target, why = top_w, low, f"Weakest set was {mn} reps (<{low}). Hold the weight and build reps."
    else:
        nxt_w, target, why = top_w, min(mn + 1, high), "Inside the rep range. Hold the weight and add a rep."
    return {
        "Last session": last_day.date(),
        "Last working sets": f"{top_w:g} kg x {', '.join(map(str, reps))}",
        "Next weight (kg)": float(nxt_w),
        "Target reps per set": int(target),
        "Why": why,
    }


def bodyweight_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Best reps per session for bodyweight moves (pull-ups, dips), unloaded sets only."""
    b = df[df["is_bodyweight"] & ~df["is_warmup"] & df["Reps"].notna() & (df["Reps"] > 0)].copy()
    b["load"] = b["Weight"].fillna(0)
    b = b[b["load"] == 0]
    return b.groupby(["Exercise", "day"]).agg(best_reps=("Reps", "max"), total_reps=("Reps", "sum"),
                                             sets=("Reps", "size")).reset_index()


def timed_progress(df: pd.DataFrame) -> pd.DataFrame:
    t = df[df["is_timed"] & df["seconds"].notna()]
    return t.groupby(["Exercise", "day"]).agg(best_seconds=("seconds", "max")).reset_index()


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    w = working_sets(df)
    vol = w.groupby("session")["volume"].sum()
    s = df.groupby("session").agg(Date=("Date", "min"), Title=("Title", "first"),
                                  Duration=("Duration", "first"),
                                  Exercises=("Exercise", "nunique")).reset_index()
    s["Volume (kg)"] = s["session"].map(vol).fillna(0).round(0)
    return s.sort_values("Date", ascending=False).drop(columns="session")


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    d = df.drop_duplicates("session")
    c = d.groupby("week").size()
    full = pd.date_range(c.index.min(), c.index.max(), freq="7D")
    return c.reindex(full, fill_value=0).rename("sessions").rename_axis("week").reset_index()


# ---------------------------------------------------------------- extras for the UI
# Fixed muscle -> colour-slot order. Colour follows the muscle, never its rank.
MUSCLE_ORDER = ["Chest", "Back", "Shoulders", "Biceps", "Triceps", "Legs", "Core", "Calves"]
# validated dark-surface categorical palette (slots 1-8)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
MUSCLE_COLORS = dict(zip(MUSCLE_ORDER, SERIES))


def muscle_balance(df: pd.DataFrame, days: int = 28, low: int = 10, high: int = 20) -> pd.DataFrame:
    """Average hard sets per week for each muscle over the `days` ending at the last session."""
    end = df["day"].max()
    h = hard_sets(df)
    h = h[(h["day"] > end - pd.Timedelta(days=days)) & h["muscle"].isin(MUSCLE_ORDER)]
    per_week = h.groupby("muscle").size() * 7.0 / days
    out = per_week.reindex(MUSCLE_ORDER).fillna(0).rename("sets_per_week").reset_index()
    out["status"] = np.where(out["sets_per_week"] < low, "Low",
                             np.where(out["sets_per_week"] > high, "High", "In range"))
    return out


def calendar_grid(df: pd.DataFrame):
    """Weeks x weekdays grid of session volume (kg x reps), plus hover text."""
    w = working_sets(df)
    vol = w.groupby("day")["volume"].sum()
    titles = df.drop_duplicates("day").set_index("day")["Title"]
    days = pd.date_range(df["day"].min() - pd.Timedelta(days=df["day"].min().weekday()),
                         df["day"].max(), freq="D")
    weeks = sorted({d - pd.Timedelta(days=d.weekday()) for d in days})
    z = [[np.nan] * len(weeks) for _ in range(7)]
    txt = [[""] * len(weeks) for _ in range(7)]
    for i, wk in enumerate(weeks):
        for r in range(7):
            d = wk + pd.Timedelta(days=r)
            if d in titles.index:
                z[r][i] = float(vol.get(d, 0.0)) or 1.0
                txt[r][i] = f"{d:%a %d %b %Y}<br>{titles[d]}<br>Volume: {vol.get(d, 0):,.0f}"
            else:
                txt[r][i] = f"{d:%a %d %b %Y}<br>Rest day"
    return [f"{wk:%d %b}" for wk in weeks], z, txt


def e1rm_with_prs(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    e = e1rm_by_session(df, exercise).sort_values("day").reset_index(drop=True)
    prev = e["e1rm"].cummax().shift(fill_value=-np.inf)
    e["is_pr"] = (e["e1rm"] > prev + 1e-9) & (e.index > 0)
    return e


def lift_change(df: pd.DataFrame, min_sessions: int = 3) -> pd.DataFrame:
    """% change in best-set e1RM from a lift's first logged session to its latest one."""
    w = working_sets(df)
    w = w[w["e1rm"].notna()]
    rows = []
    for ex, g in w.groupby("Exercise"):
        s = g.groupby("day")["e1rm"].max().sort_index()
        if len(s) >= min_sessions:
            rows.append({"Exercise": ex.strip(), "First": s.iloc[0], "Latest": s.iloc[-1],
                         "Change %": (s.iloc[-1] / s.iloc[0] - 1) * 100, "Sessions": len(s)})
    out = pd.DataFrame(rows)
    return out.sort_values("Change %", ascending=False) if len(out) else out


def demo_data(seed: int = 7) -> pd.DataFrame:
    """Synthetic Push/Pull/Legs history in Lyfta's export format, for people without an export."""
    rng = np.random.default_rng(seed)
    plan = {
        "Push": [("Bench Press", 60, 2.5), ("Seated Shoulder Press", 30, 1.5),
                 ("Triceps Pushdown", 30, 1.0), ("Cable Standing Fly", 15, 0.5)],
        "Pull": [("Bent Over Row", 50, 2.0), ("Bar Lateral Pulldown", 45, 1.5),
                 ("Alternate Biceps Curl", 12, 0.5), ("Pull-up", 0, 0)],
        "Legs": [("Full Squat", 70, 3.0), ("Romanian Deadlift", 60, 2.0),
                 ("Lever Seated Leg Curl", 40, 1.0), ("Standing Calf Raise", 50, 1.0)],
    }
    rows, day = [], pd.Timestamp("2026-06-01 18:00")
    for i in range(30):
        kind = ["Push", "Pull", "Legs"][i % 3]
        d = day + pd.Timedelta(days=int(i * 2.6 + rng.integers(0, 2)))
        for ex, base, inc in plan[kind]:
            # a few lifts stall on purpose so the plateau page has something to show
            stall = ex in ("Bent Over Row", "Seated Shoulder Press") and i > 14
            w = base + inc * (min(i, 14) if stall else i) / 1.5
            w = round(w / 2.5) * 2.5
            rows.append([kind, d, "01:05:00", ex, np.nan, 0 if base == 0 else w * 0.6, 8, "WARMUP_SET"])
            for s in range(3):
                reps = int(np.clip(rng.normal(9.5 if s == 0 else 8.5, 1.3), 5, 12)) if base else int(rng.integers(5, 9))
                rows.append([kind, d, "01:05:00", ex, np.nan, w, reps, "NORMAL_SET"])
    df = pd.DataFrame(rows, columns=["Title", "Date", "Duration", "Exercise", "Superset id",
                                     "Weight", "Reps", "Set Type"])
    df["Distance"] = np.nan
    df["Time"] = np.nan
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def load_demo() -> pd.DataFrame:
    import io
    buf = io.StringIO()
    demo_data().to_csv(buf, index=False)
    buf.seek(0)
    return load(buf)
