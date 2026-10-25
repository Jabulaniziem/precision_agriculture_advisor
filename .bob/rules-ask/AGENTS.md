# Ask Mode Context

This file provides guidance to agents when working with code in this repository.

## Non-obvious documentation context

- **`README_prototype.md`** (not `README.md`) is the canonical run guide. There is no `README.md`.
- **The Disease Check tab is presented as CNN-powered but is actually a filename lookup** — `FILENAME_DISEASE_MAP` in `app.py` maps known image filenames to hard-coded results. The CNN model is loaded but its predictions are not surfaced to the user in the current implementation.
- **Rainfall forecasts shown in the app are static pre-generated files**, not live Prophet inference. `rainfall_forecast.py` must be run separately to refresh them.
- **`python-dotenv` is in requirements but unused** — the OpenWeatherMap key is hard-coded, not loaded from `.env`.
- **`disease_cnn.py` comments (lines ~65–81)** contain a detailed explanation of three architecture iterations (v1/v2/v3) with the reasoning for each change — valuable reference for explaining model design decisions.
- **`irrigation_advisor.py` is intentionally rule-based** (not ML) — the file's docstring explicitly explains this is a deliberate design choice for explainability, not an oversight.
