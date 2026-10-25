# Agent Coding Rules

This file provides guidance to agents when working with code in this repository.

## Non-obvious coding requirements

- **Never call `model.predict()` without first checking `model is not None`** — `load_model()` returns `None` on missing file and silently breaks downstream code.
- **The feature engineering in `app.py` diverges from `yield_prediction_pipeline.py`**: `app.py` uses `sqrt(farm_size_ha)` for `fertilizer_intensity`; the training pipeline uses `log1p`. Preserve both as-is; fixing one without the other will break predictions.
- **Disease diagnosis is filename-based, not CNN-based.** Editing the CNN inference path in the Disease tab won't affect results — the real code path is `FILENAME_DISEASE_MAP.get(base_name)` followed by a fuzzy key scan. The CNN model is loaded but its output is unused at runtime.
- **`streamlit_folium` is not in `requirements.txt`** — don't add it to imports without also updating `requirements.txt`.
- **Session state keys must be initialised before the first widget call** (currently done at module level lines ~255–268 in `app.py`). Adding new state after widget instantiation causes `KeyError` on page reload.
- **`chatbot.py`'s `FarmChatbot`** is instantiated once in `st.session_state.bot` — `load_chatbot()` just returns that instance. Don't instantiate a second `FarmChatbot` directly; use `st.session_state.bot` or pass through `bot`.
- **Response templates in `INTENTS` use `{placeholder}` format strings** resolved via `template.format(**context)`. Adding a new response with a placeholder requires the corresponding key to exist in the `context` dict passed from `app.py`, or the chatbot will return the generic "enter your details first" fallback.
- **Weather API key is hard-coded** in `app.py` line ~841. If rotating the key, update it there directly (no `.env` wiring exists despite `python-dotenv` in dependencies).
