"""
Day 3-4 (v2): adds wind-specific features to help close the gap vs. the
persistence baseline that we saw in the first training run.
 
New vs. v1:
- wind_direction_100m encoded as sin/cos (direction is circular, 359 deg
  is next to 0 deg, not far from it)
- shorter wind lags (1h, 2h, 3h) in addition to the original set, since
  wind changes faster than solar
- a 3-hour rate-of-change feature for wind speed (captures "is it
  speeding up or slowing down")
 
Run:  python build_features_v2.py
"""
 
from pathlib import Path
 
import numpy as np
import pandas as pd
 
IN_CSV = Path("data/dataset.csv")   # <-- matches your actual file location
OUT_DIR = Path("features")
OUT_DIR.mkdir(exist_ok=True)
 
HORIZONS_HOURS = [24, 48]
LAGS_HOURS = [1, 2, 3, 6, 24, 48, 168]       # added 2h; kept the rest
ROLLING_WINDOWS = [6, 24]
 
TARGETS = ["wind_cf", "solar_cf"]
# Train on everything through 2025 (4 full years); test on 2026, the
# genuinely newest, most recent data — this is what "now" should forecast from.
TRAIN_END = "2025-12-31"
 
 
# ---------------------------------------------------------------- features
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["dayofweek"] = df["time"].dt.dayofweek
    df["monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df
 
 
def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Direction is circular (0-360 deg) — sin/cos encoding avoids a false
    jump between 359 and 0 that a raw degree value would create."""
    rad = np.deg2rad(df["wind_direction_100m"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    return df
 
 
def add_lag_and_rolling(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("location")[col]
    for lag in LAGS_HOURS:
        df[f"{col}_lag{lag}"] = g.shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"{col}_roll{w}"] = g.shift(1).rolling(w).mean()
    return df
 
 
def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """Is wind speeding up or slowing down over the last 3 hours?
    Uses wind_speed_100m (raw, lagged by 1h so it's known at prediction time)."""
    g = df.groupby("location")["wind_speed_100m"]
    speed_now = g.shift(1)     # last known speed
    speed_3h_ago = g.shift(4)  # speed 3 hours before that
    df["wind_speed_roc_3h"] = speed_now - speed_3h_ago
    return df
 
 
def add_targets(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("location")[col]
    for h in HORIZONS_HOURS:
        df[f"{col}_target_{h}h"] = g.shift(-h)
    return df
 
 
def add_baselines(df: pd.DataFrame, col: str) -> pd.DataFrame:
    for h in HORIZONS_HOURS:
        df[f"{col}_baseline_persistence_{h}h"] = df[col]
        df[f"{col}_baseline_seasonal_{h}h"] = df[f"{col}_lag168"]
    return df
 
 
def mae(y_true, y_pred):
    mask = y_true.notna() & y_pred.notna()
    return float((y_true[mask] - y_pred[mask]).abs().mean())
 
 
def rmse(y_true, y_pred):
    mask = y_true.notna() & y_pred.notna()
    return float(np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean()))
 
 
def build():
    df = pd.read_csv(IN_CSV, parse_dates=["time"])
    df = df.sort_values(["location", "time"]).reset_index(drop=True)
    df = add_time_features(df)
    df = add_wind_direction_features(df)
    df = add_rate_of_change(df)
 
    for col in TARGETS:
        df = add_lag_and_rolling(df, col)
        df = add_targets(df, col)
        df = add_baselines(df, col)
 
    train = df[df["time"] <= TRAIN_END].copy()
    test = df[df["time"] > TRAIN_END].copy()
 
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
 
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Train period: {train['time'].min()} to {train['time'].max()}")
    print(f"Test period:  {test['time'].min()} to {test['time'].max()}\n")
 
    print("Baseline scores on the TEST set (lower is better):\n")
    rows = []
    for col in TARGETS:
        for h in HORIZONS_HOURS:
            y_true = test[f"{col}_target_{h}h"]
            for base_name in ["persistence", "seasonal"]:
                y_pred = test[f"{col}_baseline_{base_name}_{h}h"]
                rows.append({
                    "target": col,
                    "horizon_h": h,
                    "baseline": base_name,
                    "MAE": round(mae(y_true, y_pred), 4),
                    "RMSE": round(rmse(y_true, y_pred), 4),
                })
    scorecard = pd.DataFrame(rows)
    print(scorecard.to_string(index=False))
    scorecard.to_csv(OUT_DIR / "baseline_scorecard.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'baseline_scorecard.csv'}")
 
 
if __name__ == "__main__":
    build()