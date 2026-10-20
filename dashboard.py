"""
Day 15-17: Streamlit dashboard for the Tamil Nadu wind + solar forecaster.

Reads green_hours_out/green_hours_48h.csv (produced by green_hours_v2.py)
and models/model_vs_baseline.csv (produced by train_model.py).

Run:  pip install streamlit plotly codecarbon
      streamlit run dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import predict_live  # the live-forecast logic from predict_live.py

GREEN_HOURS_CSV = Path("green_hours_out/green_hours_48h.csv")
MODEL_SCORES_CSV = Path("models/model_vs_baseline.csv")

st.set_page_config(page_title="TN Renewable Forecast", page_icon="🌬️", layout="wide")

# ---------------------------------------------------------------- data
def load_green_hours():
    # not cached with @st.cache_data, since the Refresh button needs every
    # click to re-read the file after predict_live.py overwrites it
    df = pd.read_csv(GREEN_HOURS_CSV, parse_dates=["time"])
    return df


def run_live_refresh():
    """Calls the same logic as `python predict_live.py`, but in-process so
    a dashboard button can trigger it without opening a terminal."""
    train = pd.read_csv(predict_live.TRAIN_CSV, parse_dates=["time"])
    solar_shapes = predict_live.build_diurnal_solar_shape(train)
    models = predict_live.load_models()
    feature_cols = predict_live.get_feature_cols(train)

    all_fc = []
    anchor_times = {}
    for location, (lat, lon) in predict_live.LOCATIONS.items():
        fc, anchor_time = predict_live.forecast_location(
            location, lat, lon, models, feature_cols, solar_shapes[location]
        )
        fc = predict_live.rank_green_hours(fc)
        all_fc.append(fc)
        anchor_times[location] = anchor_time

    result = pd.concat(all_fc, ignore_index=True)
    result.to_csv(predict_live.OUT_DIR / "green_hours_48h.csv", index=False)
    return anchor_times


@st.cache_data
def load_model_scores():
    if MODEL_SCORES_CSV.exists():
        return pd.read_csv(MODEL_SCORES_CSV)
    return None


def impact_numbers(df: pd.DataFrame, load_kw: float = 10.0) -> dict:
    top_quartile = df["green_score"].quantile(0.75)
    high_hours = df[df["green_score"] >= top_quartile]
    avg_random = df["green_score"].mean()
    avg_high = high_hours["green_score"].mean()
    uplift = 100 * (avg_high - avg_random) / avg_random if avg_random > 0 else 0
    return {
        "uplift_pct": round(uplift, 1),
        "best_window_start": high_hours["time"].min(),
        "best_window_end": high_hours["time"].max(),
        "best_hours_count": len(high_hours),
    }


# ---------------------------------------------------------------- app
title_col, button_col = st.columns([4, 1])
with title_col:
    st.title("🌬️ Tamil Nadu Wind & Solar Forecast")
with button_col:
    st.write("")  # vertical spacer to align the button with the title
    refresh_clicked = st.button("🔄 Refresh forecast", use_container_width=True)

st.caption(
    "Forecasts the next 48 hours of wind and solar output and recommends "
    "the greenest hours to run heavy loads. Output is derived from physics-based "
    "simulation (turbine power curve + solar irradiance model), forecast with "
    "machine learning trained and validated on real weather data."
)

if refresh_clicked:
    with st.spinner("Fetching live weather and regenerating the 48h forecast..."):
        try:
            anchor_times = run_live_refresh()
            st.success(
                f"Forecast refreshed — anchored at "
                f"{next(iter(anchor_times.values())).strftime('%b %d, %H:%M')} IST."
            )
        except Exception as e:
            st.error(f"Refresh failed: {e}")

try:
    data = load_green_hours()
except FileNotFoundError:
    st.error(
        f"Couldn't find {GREEN_HOURS_CSV}. Run build_dataset.py, build_features_v2.py, "
        "train_model.py, and predict_live.py first, in that order, or click Refresh forecast above."
    )
    st.stop()

st.caption(f"Forecast data last generated: {data['time'].min().strftime('%b %d, %Y %H:%M')} onward")

locations = sorted(data["location"].unique())
col_a, col_b = st.columns([1, 2])
with col_a:
    location = st.selectbox("Region", locations, index=0)

loc_df = data[data["location"] == location].sort_values("time")

# ---- top row: key numbers
impact = impact_numbers(loc_df)
c1, c2, c3 = st.columns(3)
c1.metric("Best-hours uplift vs. random hour", f"+{impact['uplift_pct']}%")
c2.metric("High-green hours (next 48h)", impact["best_hours_count"])
c3.metric(
    "Best window starts",
    impact["best_window_start"].strftime("%b %d, %H:%M"),
)

st.divider()

# ---- forecast chart
st.subheader(f"48-Hour Forecast — {location.title()}")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=loc_df["time"], y=loc_df["wind_cf_forecast"],
    name="Wind (capacity factor)", line=dict(color="#3b82f6"),
))
fig.add_trace(go.Scatter(
    x=loc_df["time"], y=loc_df["solar_cf_forecast"],
    name="Solar (capacity factor)", line=dict(color="#f59e0b"),
))
fig.add_trace(go.Scatter(
    x=loc_df["time"], y=loc_df["green_score"],
    name="Combined green score", line=dict(color="#22c55e", width=3, dash="dot"),
))
fig.update_layout(
    xaxis_title="Time", yaxis_title="Capacity factor (0-1)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=420, margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ---- green hours table / card
st.subheader("Recommended Green Hours")
level_order = {"High": 0, "Medium": 1, "Low": 2}
best = loc_df[loc_df["green_level"] == "High"].copy()
best_display = best[["time", "wind_cf_forecast", "solar_cf_forecast", "green_score"]].copy()
best_display.columns = ["Time", "Wind CF", "Solar CF", "Green Score"]
st.dataframe(best_display, use_container_width=True, hide_index=True)

st.info(
    f"💡 Example: running a 10 kW load during the high-green hours above gives a "
    f"green score about **{impact['uplift_pct']}% higher** than running it at a random hour "
    f"in this location's next 48h forecast."
)

st.divider()

# ---- model accuracy (from training step)
st.subheader("Model Accuracy vs. Naive Baseline")
scores = load_model_scores()
if scores is not None:
    display_scores = scores[["target", "horizon_h", "model_MAE", "baseline_MAE", "MAE_improvement_%"]]
    display_scores.columns = ["Target", "Horizon (h)", "Model MAE", "Baseline MAE", "Improvement (%)"]
    st.dataframe(display_scores, use_container_width=True, hide_index=True)
    st.caption(
        "Baseline = persistence (\"next value = current value\"). "
        "Positive improvement means the model beats simply repeating the last known reading."
    )
else:
    st.warning("Model score file not found — run train_model.py to generate it.")

st.divider()

# ---- honesty note
with st.expander("ℹ️ About the data and methodology"):
    st.markdown(
        """
        - **Training data:** Open-Meteo historical archive, 2022-2025, hourly.
        - **Live forecast:** Open-Meteo's forecast API (near-live, no multi-day
          lag), refreshed on demand via the Refresh forecast button.
        - **Power output:** simulated from weather using a physics-based turbine
          power curve (wind) and an irradiance-and-efficiency model (solar), since
          district-level real generation data was not accessible within the
          project timeline.
        - **Forecast model:** XGBoost, trained on 2022-2025 and evaluated on 2026
          (a genuinely unseen future period), compared against a persistence
          baseline.
        - **Green score:** an equal-weighted combination of forecast wind and
          solar capacity factor, ranked into High/Medium/Low relative to each
          location's own 48-hour window.
        - **Limitation:** solar's daily shape is reconstructed from each
          location's historical diurnal pattern, scaled by the model's forecast
          for that day — it reflects typical clear-sky timing more strongly than
          day-to-day cloud variation.
        """
    )

st.caption("Built for the Edunet AI Green Sustainability showcase.")