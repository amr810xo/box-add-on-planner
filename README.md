# Box Add-On Fit Planner (Streamlit)

Sections renamed for clarity:
1) **What it comes with** — set the base items and check fit.
2) **What can add on** — explore add-on combos that still fit on top of the base.

Presets:
- **Big:** 13 × 13 × 9.5 in
- **Small:** 10 × 8 × 11.5 in

Preloaded base counts (edit anytime):
- 3 × 16 oz jar, 6 × 35 oz bento, 1 × truffle box, 5 × 4 oz bottle, 1 × matcha satchet, 0 × 24 oz jar

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repo.
2. Deploy via Streamlit Community Cloud → **New App** → select `app.py`.