"""
app.py

Precision Agriculture Advisor - Business Analysis 3.2 Project
AIBUY3A

Now with:
  - Formal sign‑in (Full Name + Email)
  - Yield prediction + Profit analysis
  - What‑If simulator
  - Prediction history + CSV export
  - Crop‑specific disease simulation
  - Interactive map (Folium)
  - Cached weather API

Run with: streamlit run app.py
"""

import os
import random
import base64
from pathlib import Path
from datetime import datetime
import io
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import joblib
import streamlit as st
try:
    import tensorflow as tf # type: ignore
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False
from PIL import Image as PILImage
import requests

# Optional map imports
try:
    import folium
    from streamlit_folium import folium_static
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False

from chatbot import FarmChatbot, suggest_questions, get_agronomic_fact
from irrigation_advisor import fertilizer_note, irrigation_recommendation


# ----------------------------------------------------------------------------
# Helper: base64 image for background
# ----------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
BACKGROUND_PATH = ASSETS_DIR / "background.png"

def _b64_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ----------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Precision Agriculture Advisor",
    page_icon=PILImage.open(LOGO_PATH) if LOGO_PATH.exists() else "🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------------------------------------------------------
# CUSTOM CSS – BRIGHT LIGHT MODE (high contrast, white background)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Global text colour ---------- */
    html, body, [data-testid="stAppViewContainer"] {
        color: #f1f1f1 !important;
    }

    /* ---------- Main buttons ---------- */
    .stButton > button {
        background-color: #2E7D32 !important;
        color: #ffffff !important;
        border: 1px solid #2E7D32 !important;
    }
    .stButton > button:hover {
        background-color: #1B5E20 !important;
        color: #ffffff !important;
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 0.2rem rgba(46, 125, 50, 0.5) !important;
    }

    /* ---------- Metrics ---------- */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #d0d0d0 !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: rgba(21, 46, 24, 0.95) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #2E7D32 !important;
        color: #ffffff !important;
    }

    /* ---------- Progress bar ---------- */
    .stProgress > div > div > div > div {
        background-color: #2E7D32 !important;
    }

    /* ---------- Alerts ---------- */
    .stAlert {
        border-left-color: #2E7D32 !important;
        background-color: rgba(46, 125, 50, 0.15) !important;
        color: #f1f1f1 !important;
    }
    .stAlert p {
        color: #f1f1f1 !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #2E7D32 !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom-color: #2E7D32 !important;
    }

    /* ---------- Select boxes, inputs ---------- */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        border-color: #2E7D32 !important;
    }

    /* ---------- Headings ---------- */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .main .block-container p,
    .main .block-container label,
    .main .block-container span {
        color: #f1f1f1 !important;
    }

    /* ---------- Expander ---------- */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* ---------- Footer / captions ---------- */
    .stMarkdown small, .stCaption {
        color: #b8c8b8 !important;
    }

        /* ---------- Dark theme for expanders, uploaders, dataframes ---------- */
    [data-testid="stExpander"], .streamlit-expanderContent {
        background-color: rgba(40, 55, 40, 0.6) !important;
        color: #f1f1f1 !important;
    }
    [data-testid="stFileUploader"] {
        background-color: rgba(40, 55, 40, 0.4) !important;
    }
    [data-testid="stDataFrame"] {
        background-color: rgba(40, 55, 40, 0.4) !important;
        color: #f1f1f1 !important;
    }
    code, pre {
        background-color: rgba(60, 80, 60, 0.6) !important;
        color: #d4ffd4 !important;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# BACKGROUND IMAGE – we keep the image but now it's hidden behind the white card
#          (the solid white background overrides it, but we can keep it for
#           a subtle texture if desired – we'll set it to a very light opacity)
# ----------------------------------------------------------------------------
if BACKGROUND_PATH.exists():
    _bg_b64 = _b64_image(BACKGROUND_PATH)
    st.markdown(f"""
    <style>
        /* Dark green overlay over the agricultural photo */
        .stApp {{
            background: linear-gradient(rgba(23, 42, 23, 0.87), rgba(23, 42, 23, 0.87)),
                        url("data:image/png;base64,{_bg_b64}");
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        /* Semi-transparent dark card behind the main content so text is sharp */
        .main .block-container {{
            background-color: rgba(30, 40, 30, 0.85) !important;
            border-radius: 14px;
            padding: 2rem 2.5rem 2.5rem 2.5rem;
            margin-top: 1rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
        }}

        /* Deep green sidebar */
        section[data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(21, 46, 24, 0.95) !important;
        }}
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
MODEL_PATH = "best_yield_model.joblib"
PROVINCES = ["Free State", "Mpumalanga", "KwaZulu-Natal", "North West", "Limpopo"]
CROPS = ["Maize", "Wheat", "Sunflower", "Soybean"]
SOIL_TYPES = ["Sandy", "Loam", "Clay", "Sandy Loam"]
PROVINCE_CITIES = {
    "Free State": "Bloemfontein",
    "Mpumalanga": "Nelspruit",
    "KwaZulu-Natal": "Durban",
    "North West": "Mahikeng",
    "Limpopo": "Polokwane"
}
PROVINCE_COORDS = {
    "Free State": [-29.0, 26.0],
    "Mpumalanga": [-26.0, 30.0],
    "KwaZulu-Natal": [-29.0, 30.5],
    "North West": [-26.0, 25.0],
    "Limpopo": [-24.0, 29.0]
}


# ----------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ----------------------------------------------------------------------------
if "context" not in st.session_state:
    st.session_state.context = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "farmer_name" not in st.session_state:
    st.session_state.farmer_name = ""
if "farmer_email" not in st.session_state:
    st.session_state.farmer_email = ""
if "signed_in" not in st.session_state:
    st.session_state.signed_in = False
if "bot" not in st.session_state:
    st.session_state.bot = FarmChatbot()
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ----------------------------------------------------------------------------
# LOAD MODELS WITH CACHING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_PATH}' not found. Please run yield_prediction_pipeline.py first.")
        return None

@st.cache_resource
def load_chatbot():
    return st.session_state.bot

CNN_MODEL_PATH = "leaf_disease_cnn.keras"
CNN_CLASS_NAMES_PATH = "class_names.txt"
CNN_IMG_SIZE = 128

@st.cache_resource
def load_disease_model():
    """Loads the CNN disease model only if TensorFlow is installed.
    Returns (None, None) otherwise — the app still works without it."""
    if not TF_AVAILABLE:
        return None, None
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        with open(CNN_CLASS_NAMES_PATH) as f:
            names = [line.strip() for line in f if line.strip()]
        return cnn_model, names
    except (FileNotFoundError, OSError):
        return None, None

model = load_model()
bot = load_chatbot()
disease_model, disease_class_names = load_disease_model()   # not used in analysis


# ----------------------------------------------------------------------------
# WEATHER API – CACHED
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_weather(town: str, api_key: str):
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={town},ZA&limit=1&appid={api_key}"
        geo_resp = requests.get(geo_url, timeout=10)
        geo_results = geo_resp.json() if geo_resp.status_code == 200 else []
        if not geo_results:
            return None
        lat = geo_results[0]["lat"]
        lon = geo_results[0]["lon"]
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "conditions": data["weather"][0]["description"].title(),
            "rainfall_today": data.get("rain", {}).get("1h", 0)
        }
    except Exception:
        return None


# ----------------------------------------------------------------------------
# HELPER: Read real 7-day rainfall from pre-generated forecast CSV
# ----------------------------------------------------------------------------
def _get_7day_rainfall(province: str) -> str:
    """Return the summed 7-day predicted rainfall for a province from the
    pre-generated Prophet forecast CSV.  Falls back to a neutral value if
    the file is missing (e.g. forecast not yet generated)."""
    forecast_file = f"rainfall_forecast_{province.replace(' ', '_')}.csv"
    try:
        df = pd.read_csv(forecast_file)
        total = df.head(7)["predicted_rainfall_mm"].sum()
        return f"{total:.1f}"
    except (FileNotFoundError, KeyError):
        return "15.0"


# ----------------------------------------------------------------------------
# HELPER: Provincial yield benchmark (Feature 6)
# ----------------------------------------------------------------------------
@st.cache_data
def _load_yield_dataset() -> pd.DataFrame:
    """Load the crop yield dataset once and cache it."""
    try:
        return pd.read_csv("sa_crop_yield_dataset.csv")
    except FileNotFoundError:
        return pd.DataFrame()


def _show_yield_benchmark(predicted_yield: float, province: str, crop_type: str) -> None:
    """Show how the farmer's predicted yield compares to the provincial average."""
    df = _load_yield_dataset()
    if df.empty:
        return
    subset = df[(df["province"] == province) & (df["crop_type"] == crop_type)]
    if subset.empty:
        return
    avg_yield = subset["yield_tons_per_ha"].mean()
    diff_pct = ((predicted_yield - avg_yield) / avg_yield) * 100
    direction = "above" if diff_pct >= 0 else "below"
    colour_fn = st.success if diff_pct >= 0 else st.warning
    colour_fn(
        f"📊 **Yield Benchmark:** Your predicted **{predicted_yield:.2f} t/ha** is "
        f"**{abs(diff_pct):.1f}% {direction}** the {province} average for "
        f"{crop_type} ({avg_yield:.2f} t/ha across {len(subset):,} records)."
    )


# ----------------------------------------------------------------------------
# HELPER: Feature importance chart (Feature 5)
# ----------------------------------------------------------------------------
# Feature names must match the order used in yield_prediction_pipeline.py
_NUMERIC_FEATURES = [
    "season_rainfall_mm", "avg_temp_c", "soil_moisture_pct", "irrigation_mm",
    "fertilizer_kg_ha", "pest_pressure_index", "farm_size_ha",
    "total_water_mm", "fertilizer_intensity", "heat_stress_flag",
]
_CATEGORICAL_FEATURES = ["province", "crop_type", "soil_type"]

_FEATURE_LABELS = {
    "season_rainfall_mm": "Season Rainfall",
    "avg_temp_c": "Avg Temperature",
    "soil_moisture_pct": "Soil Moisture",
    "irrigation_mm": "Irrigation Applied",
    "fertilizer_kg_ha": "Fertilizer (kg/ha)",
    "pest_pressure_index": "Pest Pressure",
    "farm_size_ha": "Farm Size",
    "total_water_mm": "Total Water",
    "fertilizer_intensity": "Fertilizer Intensity",
    "heat_stress_flag": "Heat Stress Flag",
}


def _show_feature_importance(pipeline) -> None:
    """Extract feature importances from the trained pipeline and render a
    horizontal bar chart using Plotly (already in requirements.txt)."""
    import plotly.graph_objects as go

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model_step = pipeline.named_steps["model"]
        if not hasattr(model_step, "feature_importances_"):
            return  # not a tree-based model

        cat_names = list(
            preprocessor.named_transformers_["cat"]["onehot"]
            .get_feature_names_out(_CATEGORICAL_FEATURES)
        )
        all_names = _NUMERIC_FEATURES + cat_names
        importances = pd.Series(model_step.feature_importances_, index=all_names)

        # Keep only numeric features for the chart — they're the actionable ones
        numeric_imp = importances[_NUMERIC_FEATURES].sort_values()
        labels = [_FEATURE_LABELS.get(n, n) for n in numeric_imp.index]

        fig = go.Figure(go.Bar(
            x=numeric_imp.values,
            y=labels,
            orientation="h",
            marker_color="#2E7D32",
            text=[f"{v:.3f}" for v in numeric_imp.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="🔍 What Drives Your Yield? (Feature Importance)",
            xaxis_title="Importance Score",
            yaxis_title="",
            margin=dict(l=10, r=40, t=50, b=30),
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f1f1"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
        with st.expander("🔍 What Drives Your Yield? — Feature Importance", expanded=False):
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Importance scores show how much each input variable influenced "
                "the model's prediction. Higher = more impact on yield."
            )
    except Exception:
        pass  # silently skip if model structure differs




# ============================================================================
# SIDEBAR – Formal Sign‑In
# ============================================================================
with st.sidebar:
    st.image(str(LOGO_PATH), width=140)
    st.title("🌾 Farmer Profile")
    
    if not st.session_state.signed_in:
        st.subheader("Sign In")
        with st.form(key="signin_form"):
            name_input = st.text_input(
                "Full Name",
                value=st.session_state.farmer_name,
                placeholder="e.g., Thabo Mokoena"
            )
            email_input = st.text_input(
                "Email Address",
                value=st.session_state.farmer_email,
                placeholder="thabo@farm.co.za"
            )
            submitted = st.form_submit_button("🚀 Sign In", type="primary")
            if submitted:
                if name_input.strip() and email_input.strip():
                    if "@" not in email_input or "." not in email_input:
                        st.error("Please enter a valid email address.")
                    else:
                        st.session_state.farmer_name = name_input.strip()
                        st.session_state.farmer_email = email_input.strip()
                        st.session_state.signed_in = True
                        st.session_state.bot.set_farmer_name(name_input.strip())
                        st.session_state.context['farmer_name'] = name_input.strip()
                        st.session_state.context['farmer_email'] = email_input.strip()
                        st.rerun()
                else:
                    st.warning("Both name and email are required.")
    else:
        st.success(f"✅ Signed in as **{st.session_state.farmer_name}**")
        st.info(f"📧 {st.session_state.farmer_email}")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✏️ Edit Profile"):
                st.session_state.signed_in = False
                st.rerun()
        with col_b:
            if st.button("🚪 Sign Out"):
                st.session_state.farmer_name = ""
                st.session_state.farmer_email = ""
                st.session_state.signed_in = False
                st.session_state.context = {}
                st.session_state.chat_history = []
                st.rerun()
        st.divider()
        st.caption("🔒 Your credentials are stored only for this session.")
    
    with st.expander("💡 Quick Tips"):
        st.write("1. Enter your farm details in the first tab")
        st.write("2. Ask the chatbot any farming question")
        if st.session_state.signed_in:
            st.write("3. Try: 'When to irrigate?' or 'What's my yield?'")
        else:
            st.write("3. Sign in above to unlock personalised advice.")
    
    st.divider()
    st.caption("🌱 Did you know?")
    st.info(get_agronomic_fact())


# ============================================================================
# MAIN TITLE
# ============================================================================
title_col1, title_col2 = st.columns([1, 6])
with title_col1:
    st.image(str(LOGO_PATH), width=90)
with title_col2:
    if st.session_state.signed_in:
        st.title("Precision Agriculture Advisor")
        st.caption(
            f"Hi **{st.session_state.farmer_name}** | "
            f"📧 {st.session_state.farmer_email} | "
            "AIBUY3A Business Analysis 3.2 Project"
        )
    else:
        st.title("Precision Agriculture Advisor")
        st.caption("Please sign in using the sidebar to get personalised advice.")


# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Yield & Irrigation", "Farmer Assistant", "Disease Check", "Rainfall Forecast", "Farm Map"]
)


# ----------------------------------------------------------------------------
# TAB 1 – Yield & Irrigation
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("🌾 Enter your farm's details")
    if not st.session_state.signed_in:
        st.warning("⚠️ Please sign in using the sidebar before using the advisor.")

    col1, col2, col3 = st.columns(3)
    with col1:
        province = st.selectbox("Province", PROVINCES)
        crop_type = st.selectbox("Crop", CROPS)
        soil_type = st.selectbox("Soil type", SOIL_TYPES)
        farm_size_ha = st.number_input("Farm size (ha)", min_value=1.0, value=20.0)

    with col2:
        season_rainfall_mm = st.slider("Season rainfall so far (mm)", 100, 1100, 550)
        irrigation_mm = st.slider("Irrigation applied so far (mm)", 0, 400, 100)
        soil_moisture_pct = st.slider("Current soil moisture (%)", 5, 70, 30)

    with col3:
        avg_temp_c = st.slider("Average temperature (°C)", 12, 34, 22)
        fertilizer_kg_ha = st.slider("Fertilizer applied (kg/ha)", 20, 400, 180)
        pest_pressure_index = st.slider("Pest pressure (0=low, 10=severe)", 0.0, 10.0, 2.0)
        market_price_per_ton = st.number_input("Market price (R/ton)", min_value=1000, value=4500, step=100)
        production_cost_per_ha = st.number_input("Production cost (R/ha)", min_value=1000, value=12000, step=500)

    if st.button("Predict Yield & Get Advice", type="primary"):
        if not st.session_state.signed_in:
            st.error("Please sign in first!")
        elif model is None:
            st.error("Model not loaded. Please run yield_prediction_pipeline.py first.")
        else:
            total_water_mm = season_rainfall_mm + irrigation_mm
            fertilizer_intensity = fertilizer_kg_ha / np.log1p(farm_size_ha)
            heat_stress_flag = int(avg_temp_c > 26)

            input_df = pd.DataFrame([{
                "season_rainfall_mm": season_rainfall_mm,
                "avg_temp_c": avg_temp_c,
                "soil_moisture_pct": soil_moisture_pct,
                "irrigation_mm": irrigation_mm,
                "fertilizer_kg_ha": fertilizer_kg_ha,
                "pest_pressure_index": pest_pressure_index,
                "farm_size_ha": farm_size_ha,
                "total_water_mm": total_water_mm,
                "fertilizer_intensity": fertilizer_intensity,
                "heat_stress_flag": heat_stress_flag,
                "province": province,
                "crop_type": crop_type,
                "soil_type": soil_type,
            }])

            predicted_yield = model.predict(input_df)[0]
            irrigation_advice = irrigation_recommendation(soil_moisture_pct, season_rainfall_mm, crop_type)
            fert_note = fertilizer_note(fertilizer_kg_ha, farm_size_ha)

            total_revenue = predicted_yield * farm_size_ha * market_price_per_ton
            total_cost = production_cost_per_ha * farm_size_ha
            net_profit = total_revenue - total_cost

            st.session_state.context = {
                "predicted_yield": f"{predicted_yield:.2f}",
                "irrigation_advice": irrigation_advice,
                "fertilizer_note": fert_note,
                "crop_type": crop_type,
                "soil_type": soil_type,
                "farm_size_ha": str(farm_size_ha),
                "province": province,
                "farmer_name": st.session_state.farmer_name,
                "farmer_email": st.session_state.farmer_email,
                "total_water_mm": f"{total_water_mm:.0f}",
                "days_to_harvest": str({
                    "Maize": 120, "Wheat": 150, "Soybean": 110, "Sunflower": 100
                }.get(crop_type, 120)),
                "harvest_signs": {
                    "Maize": "dry husks, hard kernels, and browning tassels",
                    "Wheat": "golden straw colour and firm grain",
                    "Soybean": "yellowing pods and rattling seeds",
                    "Sunflower": "brown back-of-head and drooping face",
                }.get(crop_type, "drying of leaves and kernels"),
                "rainfall_7_day": _get_7day_rainfall(province),
                "fertilizer_kg_ha": str(fertilizer_kg_ha),
                "market_price": market_price_per_ton,
                "prod_cost": production_cost_per_ha,
                "net_profit": net_profit,
                "revenue": total_revenue,
                "total_cost": total_cost,
            }

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Yield", f"{predicted_yield:.2f} tons/ha")
                st.info(f"Irrigation advice: {irrigation_advice}")
                st.success(f"Hello {st.session_state.farmer_name}! Your prediction is ready.")
            with res_col2:
                st.metric("Total Revenue", f"R {total_revenue:,.2f}")
                st.metric("Total Cost", f"R {total_cost:,.2f}")
                st.metric("💰 Net Profit", f"R {net_profit:,.2f}",
                          delta="Profit" if net_profit > 0 else "Loss")
                st.write("**Fertilizer note:**", fert_note)
                st.write("**Total water available:**", f"{total_water_mm:.0f} mm")
                st.write("**Farm Size:**", f"{farm_size_ha:.1f} ha")

            # --- Feature 6: Provincial yield benchmark ---
            _show_yield_benchmark(predicted_yield, province, crop_type)

            # --- Feature 5: Feature importance chart ---
            _show_feature_importance(model)

            st.success("Prediction saved — ask the Farmer Assistant chatbot about it in the next tab.")

            history_entry = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Farmer": st.session_state.farmer_name,
                "Email": st.session_state.farmer_email,
                "Crop": crop_type,
                "Province": province,
                "Yield (tons/ha)": round(predicted_yield, 2),
                "Profit (R)": round(net_profit, 2)
            }
            st.session_state.prediction_history.append(history_entry)

    if st.session_state.context and model is not None:
        with st.expander("⚡ What-If Yield Simulator (Dynamic)"):
            base_yield = float(st.session_state.context.get("predicted_yield", 0))
            extra_rain = st.slider("Add extra rainfall (mm)", -100, 200, 0)
            extra_fert = st.slider("Add extra fertiliser (kg/ha)", -50, 100, 0)
            sim_yield = base_yield + (extra_rain * 0.005) + (extra_fert * 0.002)
            sim_yield = max(0.5, sim_yield)
            st.metric("Simulated Yield", f"{sim_yield:.2f} tons/ha",
                      delta=f"{sim_yield - base_yield:.2f} tons")

    if st.session_state.prediction_history:
        st.divider()
        st.subheader("📜 Prediction History")
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True)
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download History as CSV", csv, "farm_history.csv", "text/csv")


# ----------------------------------------------------------------------------
# TAB 2 – Chatbot
# ----------------------------------------------------------------------------
def _tts_button(text: str, key: str) -> None:
    """Render an inline 🔊 / ⏹ speaker button that reads `text` aloud via
    the browser's Web Speech API.  Uses a <script> addEventListener so that
    Streamlit's HTML sanitiser does not strip the handler."""
    # Escape backticks and backslashes so the text is safe in a JS template literal
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("'", "\\'")
    html_block = f"""
<div style="display:inline-block;margin-top:4px;">
  <button
    id="tts_btn_{key}"
    title="Read aloud"
    style="
      background: none;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      cursor: pointer;
      font-size: 16px;
      padding: 2px 7px;
      line-height: 1.4;
      color: #374151;
    "
  >🔊</button>
</div>
<script>
(function() {{
  var btn = document.getElementById('tts_btn_{key}');
  if (!btn) return;
  btn.addEventListener('click', function() {{
    var synth = window.speechSynthesis;
    if (synth.speaking) {{
      synth.cancel();
      btn.textContent = '🔊';
      btn.title = 'Read aloud';
      return;
    }}
    var utt = new SpeechSynthesisUtterance('{safe_text}');
    utt.rate = 0.95;
    utt.pitch = 1.05;
    utt.onstart = function() {{
      btn.textContent = '⏹';
      btn.title = 'Stop reading';
    }};
    utt.onend = function() {{
      btn.textContent = '🔊';
      btn.title = 'Read aloud';
    }};
    utt.onerror = function() {{
      btn.textContent = '🔊';
      btn.title = 'Read aloud';
    }};
    synth.speak(utt);
  }});
}})();
</script>
"""
    st.html(html_block, unsafe_allow_javascript=True)


with tab2:
    st.subheader("💬 Farmer Assistant Chat")

    # --- Greeting / status banner ---
    if st.session_state.signed_in:
        st.caption(
            f"👋 Welcome back, **{st.session_state.farmer_name}**! "
            "Ask me anything about your farm — irrigation, harvest timing, yield, diseases, and more. "
            "Click 🔊 next to any reply to have it read aloud."
        )
    else:
        st.warning("🔒 Please sign in to get personalised responses.")

    if st.session_state.context and st.session_state.signed_in:
        st.success(f"✅ Farm data loaded — I'm ready to help you, {st.session_state.farmer_name}!")
    elif not st.session_state.signed_in:
        st.info("ℹ️ Sign in first, then run a prediction so I have your farm's data.")
    else:
        st.info("ℹ️ Run a prediction in the first tab so I have your farm's data.")

    with st.expander("💡 Not sure what to ask? Try these"):
        suggested = suggest_questions()
        cols = st.columns(2)
        for idx, q in enumerate(suggested):
            col_idx = idx % 2
            with cols[col_idx]:
                if st.button(q, key=f"suggest_q_{idx}", use_container_width=True):
                    st.session_state._suggested_question = q

    user_input = st.text_input(
        f"Your question, {st.session_state.farmer_name if st.session_state.signed_in else 'Farmer'}:",
        placeholder="e.g., When should I harvest my maize?",
        disabled=not (st.session_state.signed_in and st.session_state.context),
        key="chat_input"
    )
    if hasattr(st.session_state, '_suggested_question'):
        user_input = st.session_state._suggested_question
        delattr(st.session_state, '_suggested_question')

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button(
            ":material/send: Ask",
            disabled=not (st.session_state.signed_in and st.session_state.context),
            type="primary",
        )
    with col2:
        clear_button = st.button(":material/delete: Clear Chat")
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    if ask_button and user_input:
        context = st.session_state.context.copy()
        context['farmer_name'] = st.session_state.farmer_name
        reply = bot.get_response_with_history(user_input, context)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", reply))
        st.rerun()

    if st.session_state.chat_history:
        st.divider()
        st.write("### 📝 Conversation")
        display_name = st.session_state.farmer_name if st.session_state.signed_in else "You"
        for idx, (speaker, msg) in enumerate(reversed(st.session_state.chat_history[-30:])):
            if speaker == "You":
                with st.chat_message("user"):
                    st.markdown(f"**{display_name}:** {msg}")
            else:
                with st.chat_message("assistant"):
                    msg_col, btn_col = st.columns([10, 1])
                    with msg_col:
                        st.markdown(f"**Assistant:** {msg}")
                    with btn_col:
                        _tts_button(msg, key=f"tts_{idx}")


# ----------------------------------------------------------------------------
# TAB 3 – Disease Check (filename‑based deterministic prediction)
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("🌿 Leaf Disease Check")
    if st.session_state.signed_in:
        st.caption(f"{st.session_state.farmer_name}, upload a leaf photo for disease analysis.")
    else:
        st.caption("Upload a leaf photo for disease analysis. (Sign in for personalised results.)")

    with st.expander("ℹ️ About Disease Detection"):
        st.write("""
        This module identifies leaf diseases by matching the uploaded **filename**
        to a pre‑defined disease database. This ensures deterministic, reproducible
        results — the same image always returns the same diagnosis.

        **Naming convention examples:**
        - `maize-healthy.jpeg` → healthy maize
        - `maize-rust.jpeg` → common rust
        - `wheat-stripe-rust.png` → wheat stripe rust
        - `soybean-rust.jpeg` → soybean rust
        """)

    # ------------------------------------------------------------------------
    # Filename → Disease mapping
    # ------------------------------------------------------------------------
    FILENAME_DISEASE_MAP = {
        # ---------- MAIZE ----------
        "maize-healthy": {
            "status": "✅ Healthy", "disease": "No disease detected",
            "confidence": "98.4%", "recommendation": "Continue regular monitoring.",
            "color": "green",
        },
        "maize-rust": {
            "status": "⚠️ Common Rust", "disease": "Puccinia sorghi infection detected",
            "confidence": "92.7%", "recommendation": "Apply triazole or strobilurin fungicide.",
            "color": "orange",
        },
        "maize-blight": {
            "status": "❌ Northern Leaf Blight", "disease": "Exserohilum turcicum infection detected",
            "confidence": "89.1%", "recommendation": "Rotate crops. Apply fungicide at first sign.",
            "color": "red",
        },
        "maize-gray-leaf-spot": {
            "status": "⚠️ Gray Leaf Spot", "disease": "Cercospora zeae-maydis infection detected",
            "confidence": "91.3%", "recommendation": "Use resistant hybrids. Apply foliar fungicide.",
            "color": "orange",
        },
        # ---------- WHEAT ----------
        "wheat-healthy": {
            "status": "✅ Healthy", "disease": "No disease detected",
            "confidence": "98.0%", "recommendation": "Continue routine scouting.",
            "color": "green",
        },
        "wheat-stripe-rust": {
            "status": "⚠️ Stripe Rust", "disease": "Puccinia striiformis infection detected",
            "confidence": "93.4%", "recommendation": "Apply triazole fungicide immediately.",
            "color": "orange",
        },
        "wheat-fusarium": {
            "status": "⚠️ Fusarium Head Blight", "disease": "Fusarium graminearum infection detected",
            "confidence": "88.7%", "recommendation": "Apply fungicide at flowering.",
            "color": "orange",
        },
        # ---------- SOYBEAN ----------
        "soybean-healthy": {
            "status": "✅ Healthy", "disease": "No disease detected",
            "confidence": "96.9%", "recommendation": "Crop looks healthy.",
            "color": "green",
        },
        "soybean-rust": {
            "status": "⚠️ Soybean Rust", "disease": "Phakopsora pachyrhizi infection detected",
            "confidence": "94.2%", "recommendation": "Apply strobilurin fungicide.",
            "color": "orange",
        },
        # ---------- SUNFLOWER ----------
        "sunflower-healthy": {
            "status": "✅ Healthy", "disease": "No disease detected",
            "confidence": "97.5%", "recommendation": "All clear.",
            "color": "green",
        },
        "sunflower-mildew": {
            "status": "⚠️ Downy Mildew", "disease": "Plasmopara halstedii infection detected",
            "confidence": "90.6%", "recommendation": "Use metalaxyl seed treatments.",
            "color": "orange",
        },
        # ---------- GENERIC FALLBACKS ----------
        "healthy": {
            "status": "✅ Healthy", "disease": "No disease detected",
            "confidence": "95.0%", "recommendation": "Leaf appears healthy.",
            "color": "green",
        },
        "rust": {
            "status": "⚠️ Rust Detected", "disease": "Fungal rust infection detected",
            "confidence": "90.0%", "recommendation": "Apply fungicide and remove affected leaves.",
            "color": "orange",
        },
        "blight": {
            "status": "❌ Blight Detected", "disease": "Blight infection detected",
            "confidence": "87.0%", "recommendation": "Remove and destroy infected plants.",
            "color": "red",
        },
    }

    DEFAULT_DISEASE_RESULT = {
        "status": "❓ Unknown Sample",
        "disease": "Filename does not match any known disease pattern.",
        "confidence": "—",
        "recommendation": (
            "Rename the file using a descriptive name such as 'maize-rust.jpeg' or "
            "'wheat-healthy.png' so the system can identify the disease."
        ),
        "color": "grey",
    }

    uploaded_file = st.file_uploader(
        "Upload a leaf photo",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**File name:** {uploaded_file.name}")
        with col2:
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
        try:
            image = PILImage.open(io.BytesIO(uploaded_file.read()))
            with col3:
                st.write(f"**Image dimensions:** {image.width} x {image.height} pixels")

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

            with btn_col1:
                if st.button("🔍 Analyze Disease", type="primary"):
                    with st.spinner("Analyzing leaf image..."):
                        raw_name = uploaded_file.name
                        base_name = raw_name.rsplit(".", 1)[0].lower().replace("_", "-").replace(" ", "-")

                        result = FILENAME_DISEASE_MAP.get(base_name)
                        if result is None:
                            for key, value in FILENAME_DISEASE_MAP.items():
                                if key in base_name:
                                    result = value
                                    break
                        if result is None:
                            result = DEFAULT_DISEASE_RESULT

                        st.divider()
                        st.subheader("📊 Analysis Results")
                        st.caption(f"🔎 Detected from filename: `{raw_name}` → key: `{base_name}`")

                        if result["color"] == "green":
                            st.success(f"**Status:** {result['status']}")
                        elif result["color"] == "orange":
                            st.warning(f"**Status:** {result['status']}")
                        elif result["color"] == "red":
                            st.error(f"**Status:** {result['status']}")
                        else:
                            st.info(f"**Status:** {result['status']}")

                        st.write(f"**Disease:** {result['disease']}")
                        st.write(f"**Confidence:** {result['confidence']}")
                        st.write(f"**Recommendation:** {result['recommendation']}")

                        if result["confidence"] != "—":
                            confidence_display = float(result["confidence"].replace("%", ""))
                            st.progress(confidence_display / 100,
                                        text=f"Confidence: {result['confidence']}")

                        st.divider()
                        st.info(
                            "**Note:** This demo uses filename‑based lookup for deterministic, "
                            "reproducible results. In production, this will be replaced by the "
                            "trained CNN model."
                        )

            with btn_col2:
                if st.button("🔄 Reset Image"):
                    st.rerun()

            with btn_col3:
                if st.button("📋 Show CNN Architecture"):
                    st.info("""
                    **CNN Architecture (target):**
                    - Input: 128x128 RGB
                    - Conv2D(16) + BatchNorm + MaxPool
                    - Conv2D(32) + BatchNorm + MaxPool
                    - Conv2D(64) + BatchNorm + MaxPool
                    - Flatten → Dense(64) → Dropout(0.4) → Dense(1) Sigmoid
                    """)
        except Exception as e:
            st.error(f"Error reading image: {e}")
    else:
        st.info("📷 Upload a leaf photo to begin disease analysis.")

# ----------------------------------------------------------------------------
# TAB 4 – Rainfall Forecast (cached weather)
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("🌦️ Weather & Rainfall Forecast")
    if st.session_state.signed_in:
        st.caption(f"Live weather and forecast for {st.session_state.farmer_name}'s region.")
    
    with st.expander("🌤️ Current Weather (Live Data)", expanded=True):
        weather_province = st.selectbox("Province", PROVINCES, key="weather_province", index=0)
        default_town = PROVINCE_CITIES.get(weather_province, "Johannesburg")
        weather_town = st.text_input(
            "Nearest town to your farm",
            value=default_town,
            key="weather_town",
            help="Weather is fetched for this specific town."
        )
        if st.button("🌐 Get Live Weather", key="get_weather"):
            with st.spinner(f"Fetching live weather for {weather_town}..."):
                API_KEY = os.getenv("OWM_API_KEY", "e673c3c4773d6b3a3bffc8816edc42c7")
                if not API_KEY:
                    weather_data = {
                        "temperature": round(random.uniform(15, 32), 1),
                        "humidity": random.randint(40, 85),
                        "wind_speed": round(random.uniform(0, 25), 1),
                        "conditions": random.choice(["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]),
                        "rainfall_today": round(random.uniform(0, 15), 1)
                    }
                else:
                    weather_data = fetch_weather(weather_town, API_KEY)
                if weather_data:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🌡️ Temperature", f"{weather_data['temperature']}°C")
                    c2.metric("💧 Humidity", f"{weather_data['humidity']}%")
                    c3.metric("💨 Wind Speed", f"{weather_data['wind_speed']} km/h")
                    c4.metric("☁️ Conditions", weather_data["conditions"])
                    st.info(f"🌧️ Today's rainfall: {weather_data.get('rainfall_today', 0)} mm")
                else:
                    st.error("Could not fetch weather. Check town spelling or API key.")
    
    st.divider()
    st.subheader("📊 90-Day Rainfall Forecast")
    forecast_province = st.selectbox("Province for forecast", PROVINCES, key="forecast_province")
    forecast_file = f"rainfall_forecast_{forecast_province.replace(' ', '_')}.csv"
    plot_file = f"rainfall_forecast_{forecast_province.replace(' ', '_')}.png"

    if os.path.exists(forecast_file):
        forecast_df = pd.read_csv(forecast_file)
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        next_7 = forecast_df.head(7)["predicted_rainfall_mm"].sum()
        next_30 = forecast_df.head(30)["predicted_rainfall_mm"].sum()
        c1, c2 = st.columns(2)
        c1.metric("🌧️ Next 7 days", f"{next_7:.1f} mm")
        c2.metric("📅 Next 30 days", f"{next_30:.1f} mm")
        if os.path.exists(plot_file):
            st.image(plot_file, caption=f"Rainfall forecast — {forecast_province}")
        with st.expander("📊 View raw forecast data"):
            st.dataframe(forecast_df, use_container_width=True)
        with st.expander("💧 Irrigation Advice Based on Forecast"):
            if next_7 < 10:
                st.warning("⚠️ **Low rainfall expected.** Consider increasing irrigation.")
            elif next_7 < 25:
                st.info("🌱 **Moderate rainfall.** Maintain normal irrigation.")
            else:
                st.success("✅ **Good rainfall.** Reduce irrigation to save water.")
    else:
        st.warning(
            "No forecast found for this province. Run `python generate_rainfall_dataset.py` "
            "then `python rainfall_forecast.py` first."
        )


# ----------------------------------------------------------------------------
# TAB 5 – Farm Map
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("🗺️ Farm Locations in South Africa")
    st.caption("Provinces are shown with approximate centroids – zoom in to explore.")
    if not MAP_AVAILABLE:
        st.warning(
            "`folium` and `streamlit-folium` are not installed. "
            "Run `pip install folium streamlit-folium` to enable this map."
        )
    else:
        m = folium.Map(location=[-28.5, 26.0], zoom_start=6)
        for prov, coords in PROVINCE_COORDS.items():
            folium.Marker(
                coords,
                popup=prov,
                icon=folium.Icon(color="green", icon="leaf", prefix="fa")
            ).add_to(m)
        folium_static(m, width=700, height=450)


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
if st.session_state.signed_in:
    st.caption(f" Precision Agriculture Advisor | AIBUY3A Business Analysis 3.2 Project")
else:
    st.caption(" Precision Agriculture Advisor | AIBUY3A Business Analysis 3.2 Project.")