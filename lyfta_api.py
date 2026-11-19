"""Pull workouts straight from Lyfta with the user's personal API key.

Lyfta gives every user a free key at https://my.lyfta.app/community/api
Endpoint: GET https://my.lyfta.app/api/v1/workouts  (Bearer key, paged, 60 req/min, 5000/day).
A sandbox with the same JSON shape needs no key: /api/sandbox/v1/workouts

The JSON is converted to the same column layout as a Lyfta CSV export, so the rest of the app
does not care where the data came from. No Streamlit imports here.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://my.lyfta.app/api/v1/workouts"
SANDBOX = "https://my.lyfta.app/api/sandbox/v1/workouts"
KEY_PAGE = "https://my.lyfta.app/community/api"

# Lyfta's API sends set types as numeric ids. Only "1" (normal) is confirmed by the sandbox.
# The others are a best guess from the CSV export's set types; verify against a real account.
SET_TYPES = {"1": "NORMAL_SET", "2": "WARMUP_SET", "3": "DROP_SET", "4": "FAILURE_SET",
             "5": "LEFT_SET", "6": "RIGHT_SET"}


class LyftaError(Exception):
    """A readable error to show the user."""


def _get(url: str, key: str | None, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Repcompass"})
    if key:
        req.add_header("Authorization", f"Bearer {key.strip()}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise LyftaError("Lyfta rejected that API key. Generate a new one and paste it again.") from None
        if e.code == 429:
            raise LyftaError("Lyfta's rate limit was hit. Wait a minute and try again.") from None
        raise LyftaError(f"Lyfta returned an error ({e.code}). Try again later.") from None
    except (urllib.error.URLError, TimeoutError):
        raise LyftaError("Could not reach Lyfta. Check your connection and try again.") from None
    except json.JSONDecodeError:
        raise LyftaError("Lyfta sent a reply Repcompass could not read.") from None


def fetch_workouts(key: str | None, sandbox: bool = False, max_pages: int = 200) -> list[dict]:
    """Every workout on the account, oldest page first. `sandbox=True` returns Lyfta's sample data."""
    if not sandbox and not (key and key.strip()):
        raise LyftaError("Paste your Lyfta API key first.")
    base = SANDBOX if sandbox else BASE
    out, page = [], 1
    while page <= max_pages:
        data = _get(f"{base}?{urllib.parse.urlencode({'page': page, 'limit': 100})}", None if sandbox else key)
        out.extend(data.get("workouts") or [])
        total = int(data.get("total_pages") or 1)
        if page >= total or not data.get("workouts"):
            break
        page += 1
        time.sleep(0.25)  # stay well under 60 requests per minute
    return out


def _num(v):
    try:
        return float(v) if v not in ("", None) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _mmss(seconds) -> str | float:
    s = _num(seconds)
    if np.isnan(s) or s <= 0:
        return np.nan
    s = int(s)
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _duration(w: dict) -> str | float:
    for k in ("duration", "workout_duration", "total_duration", "total_time"):
        v = w.get(k)
        if v in (None, ""):
            continue
        if isinstance(v, str) and ":" in v:
            return v if v.count(":") == 2 else f"00:{v}"
        s = _num(v)
        if not np.isnan(s) and s > 0:
            s = int(s)
            return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return np.nan


def to_frame(workouts: list[dict], tz: str | None = None) -> tuple[pd.DataFrame, dict]:
    """API workouts -> (frame in Lyfta CSV layout, info). Lyfta's CSV uses local time and the API
    uses UTC, so dates are shifted to `tz` (the viewer's browser time zone) to line up with exports."""
    rows, unknown = [], set()
    for w in workouts:
        when = pd.to_datetime(w.get("workout_perform_date"), utc=True, errors="coerce")
        if pd.isna(when):
            continue
        if tz:
            try:
                when = when.tz_convert(tz)
            except Exception:
                pass
        when = when.tz_localize(None)
        title, dur = w.get("title") or "Workout", _duration(w)
        for ex in w.get("exercises") or []:
            name = ex.get("excercise_name") or ex.get("exercise_name")  # Lyfta's API spells it "excercise"
            if not name:
                continue
            for s in ex.get("sets") or []:
                if s.get("is_completed") is False:
                    continue
                sid = str(s.get("set_type_id") or "1")
                if sid not in SET_TYPES:
                    unknown.add(sid)
                rows.append({
                    "Title": title, "Date": when, "Duration": dur, "Exercise": name,
                    "Superset id": _num(ex.get("superset_id")),
                    "Weight": _num(s.get("weight")), "Reps": _num(s.get("reps")),
                    "Distance": _num(s.get("distance")), "Time": _mmss(s.get("duration")),
                    "Set Type": SET_TYPES.get(sid, "NORMAL_SET"),
                })
    cols = ["Title", "Date", "Duration", "Exercise", "Superset id", "Weight", "Reps", "Distance", "Time", "Set Type"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    info = {"source": "Lyfta (synced)", "unit": None, "rows": len(df), "name": "Lyfta account",
            "workouts": len(workouts), "unknown_set_types": sorted(unknown)}
    return df, info
