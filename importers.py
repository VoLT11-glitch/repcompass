"""Read workout exports from Lyfta, Hevy and Strong, and merge several files into one history.

Everything is converted to Lyfta's column layout, then analysis.prepare() adds the derived columns.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A


def _read_text(src) -> str:
    if isinstance(src, (str, Path)):
        data = Path(src).read_bytes()
    else:
        data = src.read()
        if hasattr(src, "seek"):
            src.seek(0)
    if isinstance(data, str):
        return data
    return data.decode("utf-8-sig", errors="replace")


def _read_csv(text: str) -> pd.DataFrame:
    first = text.split("\n", 1)[0]
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, na_values=["null", ""])


def detect(cols) -> str:
    c = {x.strip().lower() for x in cols}
    if "exercise_title" in c and "start_time" in c:
        return "hevy"
    if "exercise name" in c and "workout name" in c:
        return "strong"
    if "exercise" in c and "date" in c and "set type" in c:
        return "lyfta"
    return "unknown"


def _hhmmss(seconds) -> str | float:
    if pd.isna(seconds):
        return np.nan
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _mmss(seconds) -> str | float:
    if pd.isna(seconds) or float(seconds) <= 0:
        return np.nan
    s = int(float(seconds))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _from_hevy(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = raw.rename(columns=lambda c: c.strip().lower())
    wcol = next((c for c in raw.columns if c.startswith("weight")), None)
    dcol = next((c for c in raw.columns if c.startswith("distance")), None)
    unit = "lb" if wcol and ("lb" in wcol) else "kg"
    start = pd.to_datetime(raw["start_time"], format="mixed", errors="coerce")
    end = pd.to_datetime(raw.get("end_time"), format="mixed", errors="coerce") if "end_time" in raw else None
    dur = (end - start).dt.total_seconds().map(_hhmmss) if end is not None else np.nan
    stype = raw["set_type"].astype(str).str.lower() if "set_type" in raw else pd.Series("normal", index=raw.index)
    out = pd.DataFrame({
        "Title": raw["title"] if "title" in raw else "Workout",
        "Date": start,
        "Duration": dur,
        "Exercise": raw["exercise_title"],
        "Superset id": raw["superset_id"] if "superset_id" in raw else np.nan,
        "Weight": _num(raw[wcol]) if wcol else np.nan,
        "Reps": _num(raw["reps"]) if "reps" in raw else np.nan,
        "Set Type": np.where(stype.eq("warmup"), "WARMUP_SET", "NORMAL_SET"),
        "Distance": _num(raw[dcol]) if dcol else np.nan,
        "Time": _num(raw["duration_seconds"]).map(_mmss) if "duration_seconds" in raw else np.nan,
    })
    return out, {"source": "Hevy", "unit": unit}


def _strong_duration(v) -> str | float:
    """'1h 12m', '45m', '1h', '1:05:00' -> HH:MM:SS."""
    if pd.isna(v):
        return np.nan
    t = str(v).strip()
    if ":" in t:
        return t if t.count(":") == 2 else f"00:{t}"
    h = re.search(r"(\d+)\s*h", t)
    m = re.search(r"(\d+)\s*m", t)
    s = re.search(r"(\d+)\s*s", t)
    secs = (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)
    return _hhmmss(secs) if secs else np.nan


def _from_strong(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = raw.rename(columns=lambda c: c.strip())
    order = raw["Set Order"].astype(str).str.strip().str.upper() if "Set Order" in raw else pd.Series("", index=raw.index)
    dur_col = "Duration" if "Duration" in raw else ("Workout Duration" if "Workout Duration" in raw else None)
    unit = "kg"
    if "Weight Unit" in raw and raw["Weight Unit"].astype(str).str.lower().str.contains("lb").any():
        unit = "lb"
    out = pd.DataFrame({
        "Title": raw["Workout Name"],
        "Date": pd.to_datetime(raw["Date"], format="mixed", errors="coerce"),
        "Duration": raw[dur_col].map(_strong_duration) if dur_col else np.nan,
        "Exercise": raw["Exercise Name"],
        "Superset id": np.nan,
        "Weight": _num(raw["Weight"]) if "Weight" in raw else np.nan,
        "Reps": _num(raw["Reps"]) if "Reps" in raw else np.nan,
        # Strong has no set-type column; only a literal 'W' set order marks a warm-up
        "Set Type": np.where(order.eq("W"), "WARMUP_SET", "NORMAL_SET"),
        "Distance": _num(raw["Distance"]) if "Distance" in raw else np.nan,
        "Time": _num(raw["Seconds"]).map(_mmss) if "Seconds" in raw else np.nan,
    })
    return out, {"source": "Strong", "unit": unit}


def read_export(src) -> tuple[pd.DataFrame, dict]:
    """One CSV file -> (frame in Lyfta layout, info). Raises ValueError for unknown layouts."""
    text = _read_text(src)
    raw = _read_csv(text)
    kind = detect(raw.columns)
    if kind == "hevy":
        out, info = _from_hevy(raw)
    elif kind == "strong":
        out, info = _from_strong(raw)
    elif kind == "lyfta":
        raw.columns = [c.strip() for c in raw.columns]
        for c in ("Weight", "Reps", "Distance"):
            if c in raw:
                raw[c] = _num(raw[c])
        out, info = raw, {"source": "Lyfta", "unit": None}
    else:
        raise ValueError("This CSV does not look like a Lyfta, Hevy or Strong export.")
    out = out.dropna(subset=["Date", "Exercise"]).copy()
    info["rows"] = len(out)
    return out, info


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack exports. If a session (same start time and title) appears in several files,
    keep only the copy from the file listed last, so re-exporting never double-counts."""
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    tagged = []
    for i, f in enumerate(frames):
        f = f.copy()
        f["_src"] = i
        tagged.append(f)
    all_ = pd.concat(tagged, ignore_index=True)
    all_["_key"] = pd.to_datetime(all_["Date"]).dt.strftime("%Y-%m-%d %H:%M") + "|" + all_["Title"].astype(str)
    newest = all_.groupby("_key")["_src"].transform("max")
    return all_[all_["_src"] == newest].drop(columns=["_src", "_key"]).reset_index(drop=True)


def load_files(sources: list) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Read, merge and prepare one or more exports.
    Returns (prepared df, merged raw frame in Lyfta layout, per-file info)."""
    frames, infos = [], []
    for s in sources:
        f, info = read_export(s)
        info["name"] = getattr(s, "name", str(s))
        frames.append(f)
        infos.append(info)
    raw = merge_frames(frames)
    return A.prepare(raw), raw, infos
