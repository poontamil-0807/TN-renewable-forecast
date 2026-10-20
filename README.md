# AI-Powered Wind & Solar Forecasting for Tamil Nadu

An AI-powered system that forecasts wind and solar power output for the next 48 hours across three Tamil Nadu regions (Tirunelveli, Thoothukudi, Coimbatore), and recommends "Green Hours" — the best windows to run heavy electrical loads for maximum renewable energy use.

Built for the **Edunet AI Green Sustainability Showcase**.

## Team
- **Team Lead:** Poontamil.T
- **Team Members:** Anisha.H, Lakshana.G
- **Mentor:** Vijiyalakshmi
- **College:** Anna University Regional Campus, Madurai

## What it does
- Forecasts wind and solar capacity factor 24h and 48h ahead using XGBoost, trained on 2022–2025 weather data and evaluated on genuinely unseen 2026 data
- Beats a naive persistence baseline: solar by 6.1% (24h) / 10.9% (48h), wind by 0.3% (24h) / 2.2% (48h)
- Converts the forecast into ranked "Green Hours" (High/Medium/Low) and a concrete impact estimate (75–160%+ higher green score in best hours vs. random hours)
- Live Streamlit dashboard with an on-demand refresh that pulls current weather and regenerates the forecast in real time

## Why physics-simulated output
Real district-level generation data from Grid-India/TN SLDC wasn't accessible within the project timeline. Output is instead simulated from weather using a physics-based turbine power curve (wind) and an irradiance-efficiency model (solar) — disclosed transparently, with forecasting accuracy validated on genuinely unseen future data, not on reproducing the formula.

## Pipeline
```
Weather Data (Open-Meteo API)
  → Physics-Based Simulation (wind/solar power)
  → Feature Engineering (lags, rolling averages, time + wind-direction encoding)
  → XGBoost Forecasting Models (24h & 48h, per location)
  → Green Hours Ranking (combined wind + solar green score)
  → Streamlit Dashboard (live forecast, recommendations, accuracy, methodology)
```

## Project structure
```
.
├── build_dataset.py       # Fetch historical weather, simulate wind/solar output
├── build_features_v2.py   # Build lag/rolling/time features, baselines, train/test split
├── trainmodel.py           # Train XGBoost models per target/horizon
├── green_hours_v2.py       # Static green-hours ranking (demo/offline run)
├── predict_live.py         # Live forecast using current weather (no retraining)
├── dashboard.py             # Streamlit app — chart, recommendations, refresh button
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install requests pandas numpy matplotlib xgboost scikit-learn streamlit plotly
```

## Running the pipeline
Run once, in order, to build the dataset and train the models:
```bash
python build_dataset.py
python build_features_v2.py
python trainmodel.py
python green_hours_v2.py
```

Then launch the live dashboard:
```bash
streamlit run dashboard.py
```
Click **Refresh forecast** in the app to pull current weather and regenerate a genuine next-48-hour prediction at any time — no need to rerun the full pipeline.

## Tech stack
Python, Pandas, NumPy, XGBoost, Streamlit, Plotly, Open-Meteo API

## Limitations & future work
- Validate against real Grid-India / TN SLDC generation data once accessible
- Expand to more districts with a live map view
- Add Telegram/SMS alerts for upcoming Green Hours
- Integrate CodeCarbon to report the model's own energy footprint
- Incorporate real electricity demand data to refine recommendations