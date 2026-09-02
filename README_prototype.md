# Precision Agriculture Advisor — Working Prototype

## How to run

1. Make sure all files are in the same folder:
   - app.py
   - chatbot.py
   - irrigation_advisor.py
   - best_yield_model.joblib
   - requirements.txt

2. Install dependencies:
   pip install -r requirements.txt

3. Launch the prototype:
   streamlit run app.py

4. It opens in your browser (usually http://localhost:8501)

## What's working right now

- **Yield & Irrigation Predictor tab**: enter farm details, get a real
  prediction from the trained XGBoost model, plus rule-based irrigation
  and fertilizer advice.
- **Farmer Assistant tab**: TF-IDF intent-matching chatbot that answers
  questions using your actual prediction results (not canned text).

## Stubbed for now (next build steps)

- **Disease Check tab**: will accept a leaf photo and run it through a
  CNN once that model is trained.
- **Rainfall Forecast tab**: will show a Prophet time-series forecast
  with trend/seasonality decomposition once that module is built.

## For your presentation

Run through the Yield Predictor first, screenshot the result, then
switch to the chatbot tab and ask it "what will my yield be" and "when
should I irrigate" to show the two modules are actually connected —
not just two separate demos.
