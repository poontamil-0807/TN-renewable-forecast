"""
Day 1-2: Build the training table for the Tamil Nadu wind + solar forecaster.

What it does
1. Downloads hourly historical weather from Open-Meteo (free, no API key)
   for three locations, one year at a time (avoids rate limits).
2. Converts weather into simulated power output using physics:
   - Wind: turbine power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s)
   - Solar: irradiance x area x efficiency x performance ratio, with a
     small temperature derating
3. Saves everything to data/dataset.csv and writes quick sanity-check plots.

Install:  pip install requests pandas numpy matplotlib
Run:      python build_dataset.py
"""

import os
import time

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------
# Settings (approximate wind-farm-area coordinates; verify on Google Maps)
# ----------------------------------------------------------------------
LOCATIONS = {
    "tirunelveli": (8.25, 77.55),   # Muppandal / Panagudi wind belt
    "thoothukudi": (8.76, 78.13),
    "coimbatore": (10.60, 77.25),   # Palghat gap side (Udumalpet/Palladam)
}

START_YEAR = 2022
END_YEAR = 2026  # 2026 will only fetch up to a few days before today — that's expected

HOURLY_VARS = [
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Turbine / panel assumptions (mention these in your pitch)
TURBINE_RATED_MW = 2.0
CUT_IN, RATED_SPEED, CUT_OUT = 3.0, 12.0, 25.0

PANEL_AREA_M2 = 1000.0        # 1000 m2 of panels (~190 kW peak)
PANEL_EFFICIENCY = 0.19
PERFORMANCE_RATIO = 0.80
TEMP_COEFF = -0.004           # -0.4% power per deg C above 25 C

DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "raw")


# ----------------------------------------------------------------------
# Data download
# ----------------------------------------------------------------------
def fetch_year(name: str, lat: float, lon: float, year: int) -> pd.DataFrame:
    """Fetch one year of hourly weather; cached on disk so reruns are fast.
    For the current (in-progress) year, the end date is capped a few days
    before today, since the archive API only serves completed days and
    rejects a request for dates that haven't happened yet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{name}_{year}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["time"])

    today = pd.Timestamp.now().normalize()
    safe_end = today - pd.Timedelta(days=3)  # archive API lags a few days behind "now"
    year_end = pd.Timestamp(f"{year}-12-31")
    end_date = min(year_end, safe_end)
    start_date = pd.Timestamp(f"{year}-01-01")

    if start_date > end_date:
        raise RuntimeError(
            f"No archive data available yet for {name} {year} "
            f"(latest available date is {end_date.date()})"
        )

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
    }

    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=90)
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            df["location"] = name
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:  # network errors, rate limits, etc.
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt + 1} for {name} {year} in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {name} {year}: {last_err}")


def fetch_all() -> pd.DataFrame:
    frames = []
    for name, (lat, lon) in LOCATIONS.items():
        for year in range(START_YEAR, END_YEAR + 1):
            print(f"Fetching {name} {year} ...")
            frames.append(fetch_year(name, lat, lon, year))
            time.sleep(1)  # be polite to the free API
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Physics-based targets
# ----------------------------------------------------------------------
def wind_power_mw(v: pd.Series) -> pd.Series:
    """Turbine power curve. v = wind speed at hub height (m/s)."""
    v = v.to_numpy(dtype=float)
    power = np.zeros_like(v)

    ramp = (v >= CUT_IN) & (v < RATED_SPEED)
    power[ramp] = TURBINE_RATED_MW * (
        (v[ramp] ** 3 - CUT_IN ** 3) / (RATED_SPEED ** 3 - CUT_IN ** 3)
    )

    rated = (v >= RATED_SPEED) & (v < CUT_OUT)
    power[rated] = TURBINE_RATED_MW

    # v < CUT_IN or v >= CUT_OUT stays 0
    return pd.Series(power)


def solar_power_kw(irradiance: pd.Series, temp_c: pd.Series) -> pd.Series:
    """PV output from shortwave radiation (W/m2) with temperature derating."""
    irr = irradiance.to_numpy(dtype=float)
    temp = temp_c.to_numpy(dtype=float)
    derate = 1 + TEMP_COEFF * np.maximum(temp - 25.0, 0.0)
    power = PANEL_AREA_M2 * (irr / 1000.0) * PANEL_EFFICIENCY * PERFORMANCE_RATIO * derate
    return pd.Series(np.maximum(power, 0.0))


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wind_power_mw"] = wind_power_mw(df["wind_speed_100m"]).values
    df["solar_power_kw"] = solar_power_kw(df["shortwave_radiation"], df["temperature_2m"]).values
    # Normalised 0-1 outputs make wind and solar easy to combine later
    df["wind_cf"] = df["wind_power_mw"] / TURBINE_RATED_MW
    peak_kw = PANEL_AREA_M2 * PANEL_EFFICIENCY * PERFORMANCE_RATIO  # at 1000 W/m2
    df["solar_cf"] = df["solar_power_kw"] / peak_kw
    return df


# ----------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------
def sanity_report(df: pd.DataFrame) -> None:
    print("\n--- Sanity report ---")
    print("Rows:", len(df))
    print("Missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string() or "  none")

    # Expected: 3 locations x 3 years x ~8760 hours
    print("\nRows per location:")
    print(df.groupby("location").size().to_string())

    print("\nMean capacity factor by location (wind / solar):")
    print(df.groupby("location")[["wind_cf", "solar_cf"]].mean().round(3).to_string())

    df = df.assign(month=df["time"].dt.month)
    print("\nMean wind capacity factor by month (look for a monsoon peak):")
    print(df.pivot_table(index="month", columns="location", values="wind_cf").round(2).to_string())


def make_plots(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Wind speed vs power: should trace the power curve
    fig, ax = plt.subplots(figsize=(6, 4))
    sample = df.sample(min(5000, len(df)), random_state=0)
    ax.scatter(sample["wind_speed_100m"], sample["wind_power_mw"], s=4, alpha=0.4)
    ax.set_xlabel("Wind speed at 100 m (m/s)")
    ax.set_ylabel("Simulated power (MW)")
    ax.set_title("Turbine power curve check")
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "check_power_curve.png"), dpi=130)
    plt.close(fig)

    # 2) Average wind by month per location: shows the monsoon effect
    monthly = (
        df.assign(month=df["time"].dt.month)
        .pivot_table(index="month", columns="location", values="wind_cf")
    )
    ax = monthly.plot(marker="o", figsize=(7, 4))
    ax.set_ylabel("Mean wind capacity factor")
    ax.set_title("Wind output by month (monsoon pattern)")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(DATA_DIR, "check_monthly_wind.png"), dpi=130)
    plt.close(ax.figure)

    # 3) Average solar profile by hour of day
    hourly = (
        df.assign(hour=df["time"].dt.hour)
        .pivot_table(index="hour", columns="location", values="solar_cf")
    )
    ax = hourly.plot(marker="o", figsize=(7, 4))
    ax.set_ylabel("Mean solar capacity factor")
    ax.set_title("Solar output by hour of day")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(DATA_DIR, "check_hourly_solar.png"), dpi=130)
    plt.close(ax.figure)

    print(f"\nPlots saved in ./{DATA_DIR}/")


# ----------------------------------------------------------------------
def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    weather = fetch_all()
    dataset = add_targets(weather).sort_values(["location", "time"]).reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "dataset.csv")
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  shape={dataset.shape}")

    sanity_report(dataset)
    make_plots(dataset)


if __name__ == "__main__":
    main()
