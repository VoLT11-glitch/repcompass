# Repcompass

Turn a workout export into plain-English answers: are you getting stronger, which lifts have stalled,
are you training every muscle enough, and what to lift next session.

Works with **Lyfta**, **Hevy** and **Strong** CSV exports. Upload several files to combine them.

## What it does

- Overview: training calendar, muscle balance, lift-by-lift weekly volume, neglected lifts
- Lift progress: estimated 1RM, PR markers, rep records
- Plateaus & PRs: deload check, stalled lifts, PR feed with gains
- Next workout: a full-session plan plus per-lift targets (double progression)
- Strength & goals: strength levels vs bodyweight, goal tracking with projected dates
- Compare: any two date ranges side by side
- Coach & share: weekly review, copyable AI prompt, shareable PNG card, monthly PDF report
- Data & settings: combine exports, fix muscle groups, save and restore your setup

## Run it

```
pip install -r requirements.txt
python -m streamlit run app.py
```

Drop a CSV named `export*.csv` next to `app.py` and it loads automatically, or use the sidebar uploader.
No data yet? Switch on **Use demo data**.

## Files

- `app.py` interface
- `analysis.py` core maths (e1RM, PRs, plateaus, double progression)
- `features.py` deload, strength levels, goals, comparisons, session plan, weekly review
- `importers.py` Lyfta / Hevy / Strong readers and file merging
- `report.py` share card (PNG) and monthly report (PDF)

Your file is processed in memory and is not saved by the app.
