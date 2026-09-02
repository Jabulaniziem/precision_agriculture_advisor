"""
app.py

Precision Agriculture Advisor - Working Prototype
AIBUY3A Business Analysis 3.2 Project

Run with:
    streamlit run app.py

Ties together:
- Trained XGBoost yield prediction model (yield_prediction_pipeline.py)
- Rule-based irrigation & fertilizer advisor (irrigation_advisor.py)
- TF-IDF intent chatbot (chatbot.py)

Disease detection (CNN) and rainfall time-series forecast (Prophet) are
separate modules to be wired into the "Disease Check" and "Forecast"
tabs once built - stubbed here so the prototype demo flows end-to-end.
"""

import joblib
import pandas as pd
import streamlit as st

from chatbot import FarmChatbot
from irrigation_advisor import fertilizer_note, irrigation_recommendation

st.set_page_config(page_title="Precision Agriculture Advisor", page_icon="🌾", layout="wide")

MODEL_PATH = "best_yield_model.joblib"

PROVINCES = ["Free State", "Mpumalanga", "KwaZulu-Natal", "North West", "Limpopo"]
CROPS = ["Maize", "Wheat", "Sunflower", "Soybean"]
SOIL_TYPES = ["Sandy", "Loam", "Clay", "Sandy Loam"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_chatbot():
    return FarmChatbot()


model = load_model()
bot = load_chatbot()

st.title("🌾 Precision Agriculture Advisor")
st.caption("AI Solution for Industries — Agriculture | AIBUY3A Business Analysis 3.2 Project")

if "context" not in st.session_state:
    st.session_state.context = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Yield & Irrigation Predictor", "💬 Farmer Assistant", "🖼️ Disease Check", "📈 Rainfall Forecast"]
)

# ---------------------------------------------------------------------------
# TAB 1: Yield & Irrigation Predictor
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Enter your farm's details")

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

    if st.button("Predict Yield & Get Advice", type="primary"):
        total_water_mm = season_rainfall_mm + irrigation_mm
        fertilizer_intensity = fertilizer_kg_ha / max(1.0, (farm_size_ha ** 0.5))
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

        st.session_state.context = {
            "predicted_yield": f"{predicted_yield:.2f}",
            "irrigation_advice": irrigation_advice,
            "fertilizer_note": fert_note,
        }

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Yield", f"{predicted_yield:.2f} tons/ha")
            st.info(f"💧 Irrigation advice: {irrigation_advice}")
        with res_col2:
            st.write("**Fertilizer note:**", fert_note)
            st.write("**Total water available:**", f"{total_water_mm:.0f} mm")

        st.success("Prediction saved — ask the Farmer Assistant chatbot about it in the next tab.")

# ---------------------------------------------------------------------------
# TAB 2: Chatbot
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Ask the Farmer Assistant")
    st.caption("Try: \"when should I irrigate\", \"what will my yield be\", "
               "\"how much fertilizer am I using\"")

    if not st.session_state.context:
        st.warning("Run a prediction in the first tab so the assistant has your farm's data.")

    user_input = st.text_input("Your question")
    if st.button("Ask"):
        if user_input:
            reply = bot.respond(user_input, st.session_state.context)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Assistant", reply))

    for speaker, msg in reversed(st.session_state.chat_history):
        if speaker == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg}")

# ---------------------------------------------------------------------------
# TAB 3: Disease Check (stub)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Leaf Disease Check")
    st.info("CNN disease-detection module not yet trained — this tab will accept a leaf "
            "photo upload and return a healthy/diseased classification once built.")
    st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"], disabled=True)

# ---------------------------------------------------------------------------
# TAB 4: Rainfall Forecast (stub)
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Seasonal Rainfall Forecast")
    st.info("Time-series forecasting module (Prophet) not yet wired in — this tab will show "
            "a rainfall/temperature forecast with trend and seasonality decomposition.")
