# Changelog

All notable changes to **Precision Agriculture Advisor** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2025

### Added

#### 🎤 Text-to-Speech (TTS) Speaker Button — `app.py`
- Every assistant reply in the Farmer Assistant chat now has a **🔊 speaker button**
  rendered alongside it.
- Clicking 🔊 reads the reply aloud using the browser's built-in
  **Web Speech API** (`window.speechSynthesis`) — no extra packages required.
- While speaking the icon changes to **⏹**; clicking it again cancels speech
  immediately.
- Implemented via `st.html(unsafe_allow_javascript=True)` with a `<script>`
  `addEventListener` block (inline `onclick=` handlers are stripped by
  Streamlit's HTML sanitiser and were replaced to fix a silent-button bug).
- Speech is delivered at a natural pace (`rate = 0.95`, `pitch = 1.05`).

#### 💬 Friendlier Chatbot UI — `app.py`
- Chat history now renders with native **`st.chat_message`** bubbles (user /
  assistant) instead of plain `st.markdown` lines, giving a proper
  conversational look.
- Greeting banner updated to first-person voice:  
  *"👋 Welcome back, Name! Ask me anything…"* — and explicitly tells the user
  about the 🔊 button.
- Status banners use first-person phrasing ("I'm ready to help you",
  "I have your farm's data").
- **Send** button shows a Material Symbols send icon (`:material/send: Ask`).
- **Clear Chat** button shows a delete icon (`:material/delete: Clear Chat`).
- Suggested-questions expander retitled to *"💡 Not sure what to ask? Try these"*.
- Input placeholder updated to *"e.g., When should I harvest my maize?"*.
- Section heading changed from *"📝 Conversation History"* to *"📝 Conversation"*.

#### 🌾 Harvest Intent — `chatbot.py`
- **Keyword shortcut** added to `FarmChatbot.match_intent()`: any message
  containing the word `"harvest"` is immediately routed to `harvest_advice`
  with confidence score `1.0`, bypassing the TF-IDF cosine threshold entirely.
  This makes the intent match reliably for bare phrases like *"harvest"*,
  *"when to harvest"*, *"harvest my maize"*, *"harvest schedule"*, etc.
- `_KEYWORD_INTENTS` class-level dict introduced for easy extension — add
  more keyword → intent mappings there without touching `match_intent` logic.
- **10 new example phrases** added to `harvest_advice["examples"]`:
  `"harvest"`, `"when can I harvest"`, `"harvest schedule"`,
  `"best time to harvest"`, `"harvest my crop"`, `"ready to harvest"`,
  `"harvesting time"`, `"when do I harvest"`, `"crop harvest date"`,
  `"when is harvest"`.
- **Response template rewritten** to use the natural *"you can harvest…"*
  phrasing requested:
  > *"Great news, {farmer_name}! Based on your {crop_type} crop, you can
  > harvest in approximately {days_to_harvest} days from planting. Look out
  > for {harvest_signs} as clear indicators that your crop is ready. For
  > example, you can harvest once those signs appear — typically around
  > {days_to_harvest} days into the growing season. Plan your equipment and
  > storage ahead of time to avoid post-harvest losses."*

---

## [1.0.0] — Prior release (`feat: major update`)

### Added
- **Yield Prediction tab** — XGBoost pipeline (`yield_prediction_pipeline.py`)
  trained on synthetic South African farm data.  
  Feature engineering: `total_water_mm`, `fertilizer_intensity` (log1p),
  `heat_stress_flag`.  Test R² = 0.929, RMSE = 0.437 t/ha.
- **Farmer Assistant chatbot tab** — TF-IDF cosine-similarity intent matcher
  (`chatbot.py`) covering irrigation, yield, fertilizer, disease, pest control,
  harvest, planting, soil, weather, market price, climate adaptation, and more.
- **Disease Check tab** — deterministic filename-based disease lookup
  (`FILENAME_DISEASE_MAP`) covering Maize, Wheat, Soybean, and Sunflower leaf
  diseases. CNN model (`leaf_disease_cnn.keras`) loaded optionally.
- **Rainfall Forecast tab** — Prophet-generated provincial rainfall CSVs for
  Free State, Mpumalanga, KwaZulu-Natal, North West, and Limpopo.
- **Farm Map tab** — Folium/Streamlit-Folium interactive map (optional;
  gated by `MAP_AVAILABLE` flag).
- **Prediction History** — per-session history table with CSV download.
- **Provincial yield benchmark** chart comparing predicted yield to provincial
  averages.
- **Feature importance** chart rendered from the trained XGBoost model.
- **Sidebar sign-in** with farmer name and email; personalised responses
  throughout the app.
- **Weather widget** pulling live current conditions via OpenWeatherMap API
  (`OWM_API_KEY`).
- **`.env` / `python-dotenv`** wiring for API key management.
- **`run_app.bat`** convenience launcher that activates the venv automatically.
- `AGENTS.md` workspace rules file for AI-assisted development.

### Architecture
- Sole entry point: `app.py` (Streamlit).
- Standalone retraining scripts: `yield_prediction_pipeline.py`,
  `disease_cnn.py`, `rainfall_forecast.py`.
- `chatbot.py` and `irrigation_advisor.py` are pure-Python helper modules
  imported by `app.py`.
- 5 South African provinces: Free State, Mpumalanga, KwaZulu-Natal,
  North West, Limpopo.
- Crops: Maize, Wheat, Sunflower, Soybean.
- Soil types: Sandy, Loam, Clay, Sandy Loam.

---

*Generated by IBM Bob — AI coding assistant.*
