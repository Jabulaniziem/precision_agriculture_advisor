"""
rainfall_forecast.py

Prophet-based seasonal rainfall forecast for the Precision Agriculture
Advisor. Trains one Prophet model per province on daily rainfall
history and forecasts the next 90 days - enough to inform planting/
irrigation planning for the upcoming season.

Produces:
- rainfall_forecast_<province>.png (forecast with confidence interval)
- rainfall_components_<province>.png (trend/weekly/yearly decomposition)
- a saved forecast CSV per province for the Streamlit app to read
"""

import warnings

import pandas as pd
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly

warnings.filterwarnings("ignore")

DATA_PATH = "sa_rainfall_history.csv"
FORECAST_DAYS = 90


def load_province_series(df: pd.DataFrame, province: str) -> pd.DataFrame:
    """Prophet requires columns named exactly 'ds' (date) and 'y' (value)."""
    prov_df = df[df["province"] == province][["date", "rainfall_mm"]].copy()
    prov_df.columns = ["ds", "y"]
    prov_df["ds"] = pd.to_datetime(prov_df["ds"])
    return prov_df


def train_and_forecast(prov_df: pd.DataFrame, province: str):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # rainfall has no meaningful weekly pattern
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # rainfall variance scales with season
        interval_width=0.80,
    )
    model.fit(prov_df)

    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    # Rainfall can't be negative - clip the forecast and its interval
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)

    return model, forecast


def save_static_plots(model, forecast, province: str):
    """Save matplotlib versions (works headless, no browser needed)."""
    fig1 = model.plot(forecast)
    fig1.gca().set_title(f"Rainfall Forecast — {province}")
    fig1.gca().set_ylabel("Rainfall (mm/day)")
    fig1.savefig(f"rainfall_forecast_{province.replace(' ', '_')}.png", dpi=150)

    fig2 = model.plot_components(forecast)
    fig2.savefig(f"rainfall_components_{province.replace(' ', '_')}.png", dpi=150)

    print(f"  Saved plots for {province}")


def main():
    df = pd.read_csv(DATA_PATH)
    provinces = df["province"].unique()

    summary_rows = []

    for province in provinces:
        print(f"\nTraining Prophet model for {province}...")
        prov_df = load_province_series(df, province)
        model, forecast = train_and_forecast(prov_df, province)

        save_static_plots(model, forecast, province)

        # Save the forward-looking forecast only (next FORECAST_DAYS)
        future_only = forecast.tail(FORECAST_DAYS)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        future_only.columns = ["date", "predicted_rainfall_mm", "lower_bound_mm", "upper_bound_mm"]
        future_only.to_csv(f"rainfall_forecast_{province.replace(' ', '_')}.csv", index=False)

        next_30_total = future_only.head(30)["predicted_rainfall_mm"].sum()
        summary_rows.append({"province": province, "next_30_day_total_mm": round(next_30_total, 1)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("rainfall_forecast_summary.csv", index=False)
    print("\n=== Next 30-day rainfall forecast summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
