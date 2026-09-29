# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project

Streamlit data science app — Python only, no frontend build step, no test suite.
Sole entry point: `app.py`. All other `.py` files are standalone scripts.

## Commands

```bash
# Run the app (activate venv first)
.venv\Scripts\activate          # Windows
streamlit run app.py

# Convenience launcher (Windows only, activates venv automatically)
run_app.bat

# Retrain yield model (overwrites best_yield_model.joblib)
python yield_prediction_pipeline.py

# Retrain CNN disease model (overwrites leaf_disease_cnn.keras + class_names.txt)
python disease_cnn.py

# Regenerate rainfall forecast CSVs and PNGs (read by app at runtime)
python rainfall_forecast.py
```

No linter, formatter, or test runner is configured.

## Critical Architecture Notes

- **`best_yield_model.joblib`** must exist before launching the app — run `yield_prediction_pipeline.py` to produce it. The app calls `st.error()` and returns `None` if it's missing; all yield prediction tabs then silently break.
- **`leaf_disease_cnn.keras` + `class_names.txt`** are optional — `load_disease_model()` returns `(None, None)` gracefully if TensorFlow is absent or the file is missing.
- **Disease tab uses filename-based lookup, not the CNN at runtime.** `FILENAME_DISEASE_MAP` in `app.py` (line ~649) maps uploaded image filenames to hard-coded disease results. The actual CNN model is loaded but not called for diagnosis.
- **Rainfall forecasts** are pre-generated CSVs (`rainfall_forecast_<Province>.csv`). The app reads these files directly — it does not call Prophet at runtime.
- **Weather API key** is loaded from `.env` via `python-dotenv` (`OWM_API_KEY`), with the actual key as the `os.getenv` fallback. The `.env` file is gitignored.
- **`folium` / `streamlit-folium`** are optional; `MAP_AVAILABLE` flag gates the map tab. Not in `requirements.txt` — must be installed separately.

## Feature Engineering Contract

`app.py` constructs the input DataFrame for yield prediction with these derived columns that **must match** what the trained pipeline expects:

```python
total_water_mm = season_rainfall_mm + irrigation_mm
fertilizer_intensity = fertilizer_kg_ha / np.log1p(farm_size_ha)
heat_stress_flag = int(avg_temp_c > 26)
```

> Both pipeline and `app.py` now use `np.log1p(farm_size_ha)`. Model was retrained after this fix (XGBoost, Test R² = 0.929, RMSE = 0.437 t/ha).

## Code Style

- Module-level constants in `UPPER_SNAKE_CASE`; helper functions in `snake_case`
- Optional imports wrapped in `try/except ImportError` with a boolean flag (`TF_AVAILABLE`, `MAP_AVAILABLE`)
- `@st.cache_resource` for model objects; `@st.cache_data(ttl=300)` for API calls
- All files carry a module docstring explaining purpose and how to run
- `RANDOM_SEED = 42` used throughout ML scripts for reproducibility
- Session state keys initialised once at module level before any widget calls

## Scope

Covers 5 South African provinces only: Free State, Mpumalanga, KwaZulu-Natal, North West, Limpopo.
Crops: Maize, Wheat, Sunflower, Soybean. Soil types: Sandy, Loam, Clay, Sandy Loam.
