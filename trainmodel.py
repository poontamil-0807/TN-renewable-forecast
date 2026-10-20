"""
Day 6-9: train a model for each (target, horizon) pair and compare
against the baseline scorecard from build_features.py.
 
Run:  pip install xgboost scikit-learn pandas numpy
      python train_model.py
"""
 
from pathlib import Path
 
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
 
FEAT_DIR = Path("features")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
 
TARGETS = ["wind_cf", "solar_cf"]
HORIZONS = [24, 48]
 
# Feature columns: everything that would be known at prediction time.
# We build this list dynamically to avoid typos, but explicitly EXCLUDE
# anything that is a future target or a raw future-looking column.
EXCLUDE_PREFIXES = ("wind_cf_target", "solar_cf_target",
                     "wind_cf_baseline", "solar_cf_baseline")
EXCLUDE_COLS = {"time", "location", "wind_cf", "solar_cf",
                 "wind_power_mw", "solar_power_kw",
                 "wind_speed_100m", "wind_direction_100m",
                 "temperature_2m", "cloud_cover", "shortwave_radiation"}
# Note: we exclude the RAW current-hour weather/output columns on purpose —
# at forecast time (24-48h ahead) you would only have a weather *forecast*,
# not these exact archived values, and including them risks a subtle
# leakage/optimism bias. The lag/rolling columns (shifted by >=1h) are safe
# and are what the model actually trains on.
 
 
def get_feature_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if c.startswith(EXCLUDE_PREFIXES):
            continue
        cols.append(c)
    return cols
 
 
def mae(y_true, y_pred):
    return float(np.abs(y_true - y_pred).mean())
 
 
def rmse(y_true, y_pred):
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))
 
 
def train_one(train, test, feature_cols, target, horizon):
    target_col = f"{target}_target_{horizon}h"
    base_col = f"{target}_baseline_persistence_{horizon}h"
 
    tr = train.dropna(subset=feature_cols + [target_col])
    te = test.dropna(subset=feature_cols + [target_col, base_col])
 
    X_train, y_train = tr[feature_cols], tr[target_col]
    X_test, y_test = te[feature_cols], te[target_col]
 
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
 
    pred = np.clip(model.predict(X_test), 0, 1)  # capacity factor is 0..1
    base_pred = te[base_col]
 
    result = {
        "target": target,
        "horizon_h": horizon,
        "n_train": len(tr),
        "n_test": len(te),
        "model_MAE": round(mae(y_test, pred), 4),
        "model_RMSE": round(rmse(y_test, pred), 4),
        "baseline_MAE": round(mae(y_test, base_pred), 4),
        "baseline_RMSE": round(rmse(y_test, base_pred), 4),
    }
    result["MAE_improvement_%"] = round(
        100 * (result["baseline_MAE"] - result["model_MAE"]) / result["baseline_MAE"], 1
    )
 
    model.save_model(OUT_DIR / f"{target}_{horizon}h.json")
 
    # feature importance, useful for your pitch ("what drives the forecast")
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances.sort_values(ascending=False).head(10).to_csv(
        OUT_DIR / f"{target}_{horizon}h_top_features.csv"
    )
 
    return result
 
 
def main():
    train = pd.read_csv(FEAT_DIR / "train.csv", parse_dates=["time"])
    test = pd.read_csv(FEAT_DIR / "test.csv", parse_dates=["time"])
    feature_cols = get_feature_cols(train)
 
    print(f"Using {len(feature_cols)} feature columns.\n")
 
    rows = []
    for target in TARGETS:
        for horizon in HORIZONS:
            print(f"Training {target} @ {horizon}h ...")
            rows.append(train_one(train, test, feature_cols, target, horizon))
 
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "model_vs_baseline.csv", index=False)
 
    print("\n=== Model vs. Baseline (persistence) ===\n")
    print(results.to_string(index=False))
    print(f"\nSaved {OUT_DIR / 'model_vs_baseline.csv'}")
    print(f"Saved trained models + top features in {OUT_DIR}/")
 
 
if __name__ == "__main__":
    main()
 