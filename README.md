# Repcompass

Analytics for your Lyfta workout export. Everything runs on your own computer.

## Run it
```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Then open http://localhost:8501.

## Use it
1. In Lyfta: Settings -> Export data -> CSV.
2. Drop the CSV in this folder (it picks up the newest `export*.csv`) or upload it in the sidebar.
3. No export yet? Turn on "Use demo data" in the sidebar.

## Files
- `app.py` - the interface
- `analysis.py` - all the maths (e1RM, plateaus, PRs, next-session targets)
- `.streamlit/config.toml` - dark theme
