"""
Day 10-13 (v2): fixes a bug where solar forecasts were flat near-zero
across the whole 48h window.

Root cause: the original script predicted only at now+24h and now+48h,
then drew a straight line between them. Since 24 and 48 are both
multiples of 24, those two anchor points always land on the SAME hour
of day as "now" -- so if "now" was nighttime, the entire curve was
anchored to near-zero nighttime solar.

Fix: reshape the forecast using each location's real historical diurnal
solar pattern (mean solar_cf by hour of day, from the training set),
scaled by the model's own forecasted strength for that day. Wind is
left as the same blend as before since it doesn't have this strong a
daily cycle.

Run:  python green_hours_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FEAT_DIR = Path("features")
MODEL_DIR = Path("models")
OUT_DIR = Path("green_hours_out")
OUT_DIR.mkdir(exist_ok=True)

TARGETS = ["wind_cf", "solar_cf"]
HORIZONS = [24, 48]

EXCLUDE_PREFIXES = ("wind_cf_target", "solar_cf_target",
                     "wind_cf_baseline", "solar_cf_baseline")
EXCLUDE_COLS = {"time", "location", "wind_cf", "solar_cf",
                 "wind_power_mw", "solar_power_kw",
                 "wind_speed_100m", "wind_direction_100m",
                 "temperature_2m", "cloud_cover", "shortwave_radiation"}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in EXCLUDE_COLS and not c.startswith(EXCLUDE_PREFIXES)]


def load_models():
    models = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            m = XGBRegressor()
            m.load_model(MODEL_DIR / f"{target}_{horizon}h.json")
            models[(target, horizon)] = m
    return models


def build_diurnal_solar_shape(train: pd.DataFrame) -> dict:
    """Mean solar_cf by (location, hour), normalized so the peak hour = 1.0.
    This captures the REAL daily bell curve, independent of any single
    snapshot's prediction."""
    shapes = {}
    for location, g in train.groupby("location"):
        by_hour = g.groupby(g["time"].dt.hour)["solar_cf"].mean()
        peak = by_hour.max()
        shapes[location] = (by_hour / peak) if peak > 0 else by_hour * 0
    return shapes


def forecast_for_location(test: pd.DataFrame, feature_cols: list, models: dict,
                           location: str, solar_shape: pd.Series) -> pd.DataFrame:
    loc_df = test[test["location"] == location].dropna(subset=feature_cols)
    if loc_df.empty:
        raise ValueError(f"No usable rows for {location}")
    latest = loc_df.iloc[[-1]]
    now_time = latest["time"].iloc[0]
    X = latest[feature_cols]

    preds = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            preds[(target, horizon)] = float(
                np.clip(models[(target, horizon)].predict(X)[0], 0, 1)
            )

    # Overall "how strong a solar day is this" scale factor: compare the
    # model's own predictions to what the historical shape would predict
    # for those same two anchor hours, then average the ratio. This lets
    # today's weather (cloud cover etc.) raise/lower the whole curve while
    # keeping the correct midday-peak shape.
    hour_24 = (now_time + pd.Timedelta(hours=24)).hour
    hour_48 = (now_time + pd.Timedelta(hours=48)).hour
    shape_24 = solar_shape.get(hour_24, 0.0)
    shape_48 = solar_shape.get(hour_48, 0.0)
    ratios = []
    if shape_24 > 0.05:
        ratios.append(preds[("solar_cf", 24)] / shape_24)
    if shape_48 > 0.05:
        ratios.append(preds[("solar_cf", 48)] / shape_48)
    # fall back to 1.0 (trust the historical shape as-is) if both anchors
    # were nighttime and gave no usable ratio
    solar_scale = float(np.mean(ratios)) if ratios else 1.0
    solar_scale = float(np.clip(solar_scale, 0.3, 1.7))  # keep it sane

    hours = pd.date_range(now_time + pd.Timedelta(hours=1), periods=48, freq="h")
    rows = []
    for i, t in enumerate(hours, start=1):
        frac = i / 48
        wind = preds[("wind_cf", 24)] * (1 - frac) + preds[("wind_cf", 48)] * frac
        solar = float(solar_shape.get(t.hour, 0.0)) * solar_scale
        solar = float(np.clip(solar, 0, 1))
        green_score = 0.5 * wind + 0.5 * solar
        rows.append({
            "location": location,
            "time": t,
            "hour": t.hour,
            "wind_cf_forecast": round(wind, 3),
            "solar_cf_forecast": round(solar, 3),
            "green_score": round(green_score, 3),
        })
    return pd.DataFrame(rows)


def rank_green_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    q_low, q_high = df["green_score"].quantile([0.33, 0.66])
    df["green_level"] = df["green_score"].apply(
        lambda v: "High" if v >= q_high else ("Medium" if v >= q_low else "Low")
    )
    return df


def impact_example(df: pd.DataFrame, load_kw: float = 10.0) -> dict:
    top_quartile = df["green_score"].quantile(0.75)
    high_hours = df[df["green_score"] >= top_quartile]
    avg_random = df["green_score"].mean()
    avg_high = high_hours["green_score"].mean()
    uplift_pct = 100 * (avg_high - avg_random) / avg_random if avg_random > 0 else float("nan")
    return {
        "load_kw": load_kw,
        "avg_green_score_random_hour": round(avg_random, 3),
        "avg_green_score_best_hours": round(avg_high, 3),
        "relative_uplift_%": round(uplift_pct, 1),
        "best_hours_example": high_hours["time"].dt.strftime("%Y-%m-%d %H:%M").tolist()[:5],
    }


def main():
    train = pd.read_csv(FEAT_DIR / "train.csv", parse_dates=["time"])
    test = pd.read_csv(FEAT_DIR / "test.csv", parse_dates=["time"])
    feature_cols = get_feature_cols(test)
    models = load_models()
    solar_shapes = build_diurnal_solar_shape(train)

    all_forecasts = []
    for location in test["location"].unique():
        print(f"Forecasting next 48h for {location} ...")
        fc = forecast_for_location(test, feature_cols, models, location,
                                    solar_shapes[location])
        fc = rank_green_hours(fc)
        all_forecasts.append(fc)

        # quick check: print midday vs midnight solar to confirm the fix
        midday = fc[fc["hour"].between(11, 13)]["solar_cf_forecast"].mean()
        midnight = fc[fc["hour"].isin([0, 1, 2])]["solar_cf_forecast"].mean()
        print(f"  Sanity check: avg solar_cf midday={midday:.3f} "
              f"vs midnight={midnight:.3f} (midday should be clearly higher)")

        impact = impact_example(fc)
        print(f"  Impact example ({location}): running a {impact['load_kw']}kW load "
              f"in the best hours gives a green score {impact['relative_uplift_%']}% "
              f"higher than a random hour.")
        print(f"  Best hours sample: {impact['best_hours_example']}\n")

    result = pd.concat(all_forecasts, ignore_index=True)
    result.to_csv(OUT_DIR / "green_hours_48h.csv", index=False)
    print(f"Saved {OUT_DIR / 'green_hours_48h.csv'} — this feeds the dashboard next.")


if __name__ == "__main__":
    main()