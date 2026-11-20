"""Quick checks to run before every push:  python tests/smoke_test.py  (from the app folder).

Covers: demo data, the Lyfta API sandbox conversion, saving/restoring history, merging a sync with a CSV,
and a real export if one sits in ../data or next to app.py."""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importers as I  # noqa: E402
import lyfta_api as L  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check.failed = True


check.failed = False

# 1. Lyfta API sandbox -> Lyfta layout
try:
    ws = L.fetch_workouts(None, sandbox=True)
    f, info = L.to_frame(ws, "Asia/Kolkata")
    check("sandbox returns workouts", len(ws) >= 1 and len(f) >= 1)
    check("sandbox columns match CSV layout", list(f.columns)[:4] == ["Title", "Date", "Duration", "Exercise"])
    prepared, raw, _ = I.combine([f], [info])
    check("sandbox frame prepares", len(prepared) == len(f))
except L.LyftaError as e:
    print("SKIP sandbox (offline?):", e)

# 2. bad key gives a readable error
try:
    L.fetch_workouts("not-a-real-key")
    check("bad key rejected", False)
except L.LyftaError as e:
    check("bad key rejected with message", "key" in str(e).lower() or "reach" in str(e).lower())

# 3. history pack / unpack round trip and sync replacing same day + title
exports = sorted([*ROOT.glob("export*.csv"), *(ROOT.parent / "data").glob("export*.csv")])
if exports:
    real, rinfo = I.read_export(exports[-1])
    blob = I.pack_history(real)
    back = I.unpack_history(blob)
    check(f"pack/unpack keeps all {len(real)} rows ({len(blob) // 1024} KB stored)", back is not None and len(back) == len(real))
    # a fake "sync" of the newest session with the clock shifted 5h30 must not double count
    last = real[real["Date"] == real["Date"].max()].copy()
    last["Date"] = (pd.to_datetime(last["Date"]) + pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d %H:%M:%S")
    _, merged, _ = I.combine([real, last], [rinfo, {"source": "Lyfta (synced)"}])
    check("sync replaces same session instead of duplicating", len(merged) == len(real))
else:
    print("SKIP real export (none found)")

# 3b. calendar month labels line up with the calendar's week columns (same formula app.py uses)
import analysis as A  # noqa: E402
for name, frame in [("demo", A.load_demo())] + ([("real", I.load_files([exports[-1]])[0])] if exports else []):
    wk, z, _ = A.calendar_grid(frame)
    d0, d1 = frame["day"].min(), frame["day"].max()
    starts = pd.date_range(d0 - pd.Timedelta(days=d0.weekday()), d1, freq="7D")
    check(f"calendar weeks line up ({name})", len(starts) == len(wk) == len(z[0]))

# 4. the app runs headless on demo data and on the local export
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
at.run()
check("app runs with no data (or local export)", not at.exception)
at.session_state["demo"] = True
at.run()
check("app runs on demo data", not at.exception)
if at.exception:
    print(at.exception)
check("no Streamlit warnings", not [w for w in at.warning if "Session State" in str(w.value)])
check("8 tabs", len(at.tabs) == 8)
check("settings live in the Settings tab", [r for r in at.radio if r.key == "unit"] and [n for n in at.number_input if n.key == "low"])
charts = at.get("plotly_chart")
figs = [json.loads(c.proto.spec) for c in charts]
check("charts never zoom on drag", figs and all(f["layout"].get("dragmode") is False
                                                 and f["layout"]["xaxis"].get("fixedrange") for f in figs))
low_box = [n for n in at.number_input if n.key == "low"][0]
low_box.set_value(6).run()
check("changing a setting keeps it", not at.exception and at.session_state["low"] == 6)

# 5. a first-time visitor (no local file) gets the welcome screen, and the demo button works
os.environ["REPCOMPASS_NO_LOCAL"] = "1"
w = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
w.run()
w.run()  # second run: past the "checking this browser" step
labels = [b.label for b in w.button]
check("welcome screen shows the demo button", "Try it with demo data" in labels and not w.exception)
check("welcome screen has no API key field", not [t for t in w.text_input if t.key == "lyfta_key"])
[b for b in w.button if b.label == "Try it with demo data"][0].click().run()
check("demo button opens the dashboard", not w.exception and len(w.tabs) == 8)
del os.environ["REPCOMPASS_NO_LOCAL"]

sys.exit(1 if check.failed else 0)
