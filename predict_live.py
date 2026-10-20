"""
Live prediction: run this any time to get a genuinely current 48h forecast,
without retraining anything.

Why this is separate from build_dataset.py / green_hours_v2.py:
- Those scripts work off a static CSV that's only as fresh as your last
  full pipeline run, and the archive API has a 1-3 day publishing delay.
- This script instead pulls RECENT data from Open-Meteo's forecast API
  (which covers the last few days plus the next 16 days, near-live),
  builds the same features the trained models expect, and predicts from
  the actual current moment.

Run:  python predict_live.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor

MODEL_DIR = Path("models")
TRAIN_CSV = Path("features/train.csv")   # used only for the solar diurnal shape
OUT_DIR = Path("green_hours_out")
OUT_DIR.mkdir(exist_ok=True)

LOCATIONS = {
    "tirunelveli": (8.25, 77.55),
    "thoothukudi": (8.76, 78.13),
    "coimbatore": (10.60, 77.25),
}

HOURLY_VARS = [
    "wind_speed_100m", "wind_direction_100m", "temperature_2m",
    "cloud_cover", "shortwave_radiation",
]

TARGETS = ["wind_cf", "solar_cf"]
HORIZONS = [24, 48]

WIND = dict(rated_mw=2.0, cut_in=3.0, rated_speed=12.0, cut_out=25.0)
SOLAR = dict(area_m2=1000.0, eff=0.19, pr=0.80, temp_coeff=0.004)

LAGS_HOURS = [1, 2, 3, 6, 24, 48, 168]
ROLLING_WINDOWS = [6, 24]

EXCLUDE_PREFIXES = ("wind_cf_target", "solar_cf_target",
                     "wind_cf_baseline", "solar_cf_baseline")
EXCLUDE_COLS = {"time", "location", "wind_cf", "solar_cf",
                 "wind_power_mw", "solar_power_kw",
                 "wind_speed_100m", "wind_direction_100m",
                 "temperature_2m", "cloud_cover", "shortwave_radiation"}


# ---------------------------------------------------------------- live data
def fetch_recent_and_forecast(lat: float, lon: float, retries: int = 3) -> pd.DataFrame:
    """Open-Meteo's forecast endpoint covers ~92 days of recent past plus
    16 days ahead, all near-live -- no multi-day publishing lag like the
    archive endpoint. We pull the last 10 days (covers our 168h lag) plus
    the next 2 days.

    Retries on transient network/SSL errors (common on flaky Wi-Fi or
    when antivirus HTTPS scanning interferes), with a short backoff."""
    import time

    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
        "past_days": 10,
        "forecast_days": 2,
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params=params,
                timeout=60,
            )
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except (requests.exceptions.RequestException, requests.exceptions.SSLError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(3 * attempt)  # 3s, 6s, ... backoff
    raise RuntimeError(
        f"Failed to fetch live weather for ({lat}, {lon}) after {retries} attempts: {last_err}"
    )


# ---------------------------------------------------------------- physics
def wind_power_mw(v):
    v = np.asarray(v, dtype=float)
    ramp = WIND["rated_mw"] * (v**3 - WIND["cut_in"]**3) / (WIND["rated_speed"]**3 - WIND["cut_in"]**3)
    power = np.where(v < WIND["cut_in"], 0.0, np.where(v < WIND["rated_speed"], ramp, WIND["rated_mw"]))
    return np.where(v >= WIND["cut_out"], 0.0, power)


def solar_power_kw(irradiance, temp_c):
    irradiance = np.asarray(irradiance, dtype=float)
    cell_temp = np.asarray(temp_c, dtype=float) + 0.03 * irradiance
    derate = 1.0 - SOLAR["temp_coeff"] * np.maximum(0.0, cell_temp - 25.0)
    return SOLAR["area_m2"] * (irradiance / 1000.0) * SOLAR["eff"] * SOLAR["pr"] * derate


# ---------------------------------------------------------------- features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    df["wind_power_mw"] = wind_power_mw(df["wind_speed_100m"])
    df["wind_cf"] = df["wind_power_mw"] / WIND["rated_mw"]
    df["solar_power_kw"] = solar_power_kw(df["shortwave_radiation"], df["temperature_2m"])
    solar_peak = SOLAR["area_m2"] * SOLAR["eff"] * SOLAR["pr"]
    df["solar_cf"] = df["solar_power_kw"] / solar_peak

    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["dayofweek"] = df["time"].dt.dayofweek
    df["monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    rad = np.deg2rad(df["wind_direction_100m"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)

    for col in TARGETS:
        for lag in LAGS_HOURS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll{w}"] = df[col].shift(1).rolling(w).mean()

    speed_now = df["wind_speed_100m"].shift(1)
    speed_3h_ago = df["wind_speed_100m"].shift(4)
    df["wind_speed_roc_3h"] = speed_now - speed_3h_ago

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in EXCLUDE_COLS and not c.startswith(EXCLUDE_PREFIXES)]


# ---------------------------------------------------------------- model
def load_models():
    models = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            m = XGBRegressor()
            m.load_model(MODEL_DIR / f"{target}_{horizon}h.json")
            models[(target, horizon)] = m
    return models


def build_diurnal_solar_shape(train: pd.DataFrame) -> dict:
    shapes = {}
    for location, g in train.groupby("location"):
        by_hour = g.groupby(g["time"].dt.hour)["solar_cf"].mean()
        peak = by_hour.max()
        shapes[location] = (by_hour / peak) if peak > 0 else by_hour * 0
    return shapes


def forecast_location(location, lat, lon, models, feature_cols, solar_shape):
    raw = fetch_recent_and_forecast(lat, lon)
    feat = build_features(raw)

    now = pd.Timestamp.now(tz=None).floor("h")
    # use the latest row at or before "now" that has all features populated
    available = feat[feat["time"] <= now].dropna(subset=feature_cols)
    if available.empty:
        raise RuntimeError(f"No usable feature row for {location} near {now}")
    latest = available.iloc[[-1]]
    anchor_time = latest["time"].iloc[0]
    X = latest[feature_cols]

    preds = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            preds[(target, horizon)] = float(
                np.clip(models[(target, horizon)].predict(X)[0], 0, 1)
            )

    hour_24 = (anchor_time + pd.Timedelta(hours=24)).hour
    hour_48 = (anchor_time + pd.Timedelta(hours=48)).hour
    shape_24 = solar_shape.get(hour_24, 0.0)
    shape_48 = solar_shape.get(hour_48, 0.0)
    ratios = []
    if shape_24 > 0.05:
        ratios.append(preds[("solar_cf", 24)] / shape_24)
    if shape_48 > 0.05:
        ratios.append(preds[("solar_cf", 48)] / shape_48)
    solar_scale = float(np.clip(np.mean(ratios), 0.3, 1.7)) if ratios else 1.0

    hours = pd.date_range(anchor_time + pd.Timedelta(hours=1), periods=48, freq="h")
    rows = []
    for i, t in enumerate(hours, start=1):
        frac = i / 48
        wind = preds[("wind_cf", 24)] * (1 - frac) + preds[("wind_cf", 48)] * frac
        solar = float(np.clip(solar_shape.get(t.hour, 0.0) * solar_scale, 0, 1))
        green_score = 0.5 * wind + 0.5 * solar
        rows.append({
            "location": location, "time": t, "hour": t.hour,
            "wind_cf_forecast": round(wind, 3),
            "solar_cf_forecast": round(solar, 3),
            "green_score": round(green_score, 3),
        })
    return pd.DataFrame(rows), anchor_time


def rank_green_hours(df):
    q_low, q_high = df["green_score"].quantile([0.33, 0.66])
    df = df.copy()
    df["green_level"] = df["green_score"].apply(
        lambda v: "High" if v >= q_high else ("Medium" if v >= q_low else "Low")
    )
    return df


def main():
    train = pd.read_csv(TRAIN_CSV, parse_dates=["time"])
    solar_shapes = build_diurnal_solar_shape(train)
    models = load_models()
    feature_cols = get_feature_cols(train)

    all_fc = []
    for location, (lat, lon) in LOCATIONS.items():
        print(f"Fetching live data and forecasting for {location} ...")
        fc, anchor_time = forecast_location(location, lat, lon, models, feature_cols, solar_shapes[location])
        fc = rank_green_hours(fc)
        all_fc.append(fc)
        print(f"  Anchored at: {anchor_time} -> forecasting {fc['time'].min()} to {fc['time'].max()}")

    result = pd.concat(all_fc, ignore_index=True)
    result.to_csv(OUT_DIR / "green_hours_48h.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'green_hours_48h.csv'} with current timestamps.")
    print("Re-run this script any time to refresh the dashboard's forecast.")


if __name__ == "__main__":
    main()