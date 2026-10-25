# Plan Mode Architectural Rules

This file provides guidance to agents when working with code in this repository.

## Non-obvious architectural constraints

- **Single-file Streamlit app**: all UI, model loading, and business logic live in `app.py`. There is no routing, no page directory, no multi-page Streamlit setup.
- **Feature engineering split across two files**: training (`yield_prediction_pipeline.py`) and inference (`app.py`) compute `fertilizer_intensity` differently (`log1p` vs `sqrt`). Any plan to unify them must retrain the model after changing inference-side logic.
- **Three independent ML artifacts** that must be regenerated separately:
  1. `best_yield_model.joblib` — via `yield_prediction_pipeline.py`
  2. `leaf_disease_cnn.keras` + `class_names.txt` — via `disease_cnn.py`
  3. `rainfall_forecast_<Province>.csv` (×5) — via `rainfall_forecast.py`
- **No shared utility module** — helper functions (`fertilizer_note`, `irrigation_recommendation`) live in `irrigation_advisor.py`; chatbot utilities in `chatbot.py`. Any new shared logic needs a new module or placement in one of these.
- **No test infrastructure** — no pytest config, no test directory, no fixtures. Any testing plan must start from scratch.
- **Geography is hard-coded**: provinces, cities, and coordinates are dict literals in `app.py`. Expanding coverage requires updating `PROVINCES`, `PROVINCE_CITIES`, `PROVINCE_COORDS` constants and retraining/regenerating all three artifact types above.
- **TensorFlow is a soft dependency**: the app is designed to run without it (CNN tab degrades gracefully). Plans adding TF features must preserve this isolation pattern.
