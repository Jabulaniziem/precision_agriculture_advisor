"""
generate_dataset.py

Generates a SIMULATED South African crop yield dataset for the
Precision Agriculture project (AIBUY3A - Business Analysis 3.2).

Real farm-level sensor data (soil moisture, irrigation logs) is not
publicly available at this granularity, so this script creates a
realistic synthetic dataset based on known agronomic relationships
(rainfall, temperature, soil type, fertilizer -> yield). This is a
standard and accepted approach for demonstration projects - state
clearly in your documentation that this dataset is simulated.

Provinces and crops are chosen to reflect real South African farming
regions (data.gov.za / DALRRD publications used as general reference
for realistic ranges, not as a literal data source).
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 3000

np.random.seed(RANDOM_SEED)

PROVINCES = ["Free State", "Mpumalanga", "KwaZulu-Natal", "North West", "Limpopo"]
CROPS = ["Maize", "Wheat", "Sunflower", "Soybean"]
SOIL_TYPES = ["Sandy", "Loam", "Clay", "Sandy Loam"]

# Base yield (tons/hectare) and rainfall sensitivity differ per crop
CROP_BASE_YIELD = {"Maize": 5.5, "Wheat": 3.2, "Sunflower": 1.8, "Soybean": 2.6}
CROP_RAIN_SENSITIVITY = {"Maize": 0.55, "Wheat": 0.40, "Sunflower": 0.30, "Soybean": 0.45}

SOIL_QUALITY_FACTOR = {"Sandy": 0.85, "Loam": 1.15, "Clay": 0.95, "Sandy Loam": 1.05}


def generate_row():
    province = np.random.choice(PROVINCES)
    crop = np.random.choice(CROPS)
    soil = np.random.choice(SOIL_TYPES)

    season_rainfall_mm = np.clip(np.random.normal(550, 150), 150, 1100)
    avg_temp_c = np.clip(np.random.normal(22, 3.5), 12, 34)
    fertilizer_kg_ha = np.clip(np.random.normal(180, 60), 20, 400)
    soil_moisture_pct = np.clip(
        (season_rainfall_mm / 1100) * 60 + np.random.normal(0, 5), 5, 70
    )
    irrigation_mm = np.clip(np.random.normal(120, 80), 0, 400)
    pest_pressure_index = np.clip(np.random.beta(2, 6) * 10, 0, 10)  # 0=low, 10=severe
    farm_size_ha = np.clip(np.random.exponential(25), 1, 400)

    base = CROP_BASE_YIELD[crop]
    rain_sens = CROP_RAIN_SENSITIVITY[crop]
    soil_factor = SOIL_QUALITY_FACTOR[soil]

    rain_effect = rain_sens * (season_rainfall_mm - 550) / 100
    temp_penalty = -0.08 * max(0, avg_temp_c - 26) ** 1.3
    fert_effect = 0.004 * fertilizer_kg_ha
    irrigation_effect = 0.003 * irrigation_mm
    pest_penalty = -0.12 * pest_pressure_index
    noise = np.random.normal(0, 0.35)

    yield_tons_ha = (
        base * soil_factor
        + rain_effect
        + temp_penalty
        + fert_effect
        + irrigation_effect
        + pest_penalty
        + noise
    )
    yield_tons_ha = round(max(0.1, yield_tons_ha), 2)

    return {
        "province": province,
        "crop_type": crop,
        "soil_type": soil,
        "season_rainfall_mm": round(season_rainfall_mm, 1),
        "avg_temp_c": round(avg_temp_c, 1),
        "soil_moisture_pct": round(soil_moisture_pct, 1),
        "irrigation_mm": round(irrigation_mm, 1),
        "fertilizer_kg_ha": round(fertilizer_kg_ha, 1),
        "pest_pressure_index": round(pest_pressure_index, 2),
        "farm_size_ha": round(farm_size_ha, 1),
        "yield_tons_per_ha": yield_tons_ha,
    }


def main():
    rows = [generate_row() for _ in range(N_RECORDS)]
    df = pd.DataFrame(rows)

    # Inject a small amount of missing data + a few outliers, since real
    # farm data is never perfectly clean - this also gives you something
    # genuine to clean in the data pipeline step (worth mentioning in docs).
    missing_idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_idx, "soil_moisture_pct"] = np.nan

    outlier_idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_idx, "yield_tons_per_ha"] *= 3.0  # simulate data-entry errors

    out_path = "sa_crop_yield_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records -> {out_path}")
    print(df.head())
    print("\nSummary stats:")
    print(df.describe(include="all").transpose())


if __name__ == "__main__":
    main()
