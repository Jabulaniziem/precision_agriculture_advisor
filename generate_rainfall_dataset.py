"""
generate_rainfall_dataset.py

Generates SIMULATED daily rainfall and temperature history for South
African provinces, for the Prophet time-series forecasting component.

Real historical weather data is available from the South African
Weather Service (SAWS) and would be the intended real-world data
source referenced in your Data section - state this clearly in your
documentation. This script generates synthetic daily rainfall with
realistic seasonal structure (SA has a summer-rainfall pattern in
most provinces, peaking Nov-Mar) so the forecasting pipeline below is
fully working and can be pointed at real SAWS data later by simply
swapping the data-loading step.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROVINCES = ["Free State", "Mpumalanga", "KwaZulu-Natal", "North West", "Limpopo"]
N_YEARS = 5


def generate_province_series(province: str, start_date: str, n_years: int) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=365 * n_years, freq="D")
    day_of_year = dates.dayofyear

    # Summer-rainfall seasonality: peak around day ~30 (late Jan) and
    # ~335 (early Dec), trough around day ~180 (late June/winter)
    seasonal = 1.0 + 0.9 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
    seasonal = np.clip(seasonal, 0.05, None)

    base_daily_mm = {
        "Free State": 1.6, "Mpumalanga": 2.1, "KwaZulu-Natal": 2.6,
        "North West": 1.4, "Limpopo": 1.5,
    }[province]

    # Rain is intermittent - use a probability-of-rain gate plus amount
    rain_prob = np.clip(0.15 + 0.35 * (seasonal - 1), 0.03, 0.6)
    rains_today = np.random.random(len(dates)) < rain_prob
    amounts = np.random.gamma(shape=2.0, scale=base_daily_mm * seasonal, size=len(dates))
    rainfall_mm = np.where(rains_today, amounts, 0.0)

    # Slight long-term trend (e.g. -1mm/year average, representing a mild
    # drying trend some SA regions have reported) plus noise
    trend = -0.003 * np.arange(len(dates))
    rainfall_mm = np.clip(rainfall_mm + trend, 0, None)

    avg_temp_c = (
        22 + 6 * np.cos(2 * np.pi * (day_of_year - 15) / 365 + np.pi)
        + np.random.normal(0, 1.5, len(dates))
    )

    return pd.DataFrame({
        "date": dates,
        "province": province,
        "rainfall_mm": np.round(rainfall_mm, 1),
        "avg_temp_c": np.round(avg_temp_c, 1),
    })


def main():
    all_dfs = [generate_province_series(p, "2021-01-01", N_YEARS) for p in PROVINCES]
    df = pd.concat(all_dfs, ignore_index=True)

    out_path = "sa_rainfall_history.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows across {len(PROVINCES)} provinces -> {out_path}")
    print(df.groupby("province")["rainfall_mm"].agg(["mean", "sum"]).round(1))


if __name__ == "__main__":
    main()
