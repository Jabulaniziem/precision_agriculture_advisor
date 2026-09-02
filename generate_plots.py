"""
generate_plots.py
Produces two figures for the project documentation:
1. Actual vs Predicted yield scatter plot (model accuracy visual)
2. Feature importance bar chart
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from yield_prediction_pipeline import (
    CATEGORICAL_FEATURES, DATA_PATH, NUMERIC_FEATURES, RANDOM_SEED, TARGET,
    clean_data, engineer_features, get_feature_importance, load_data,
)

plt.style.use("seaborn-v0_8-whitegrid")

df = load_data(DATA_PATH)
df = clean_data(df)
df = engineer_features(df)

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

pipeline = joblib.load("best_yield_model.joblib")
preds = pipeline.predict(X_test)

# --- Plot 1: Actual vs Predicted ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, preds, alpha=0.4, color="#2E7D32", edgecolor="none")
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual Yield (tons/ha)")
ax.set_ylabel("Predicted Yield (tons/ha)")
ax.set_title("XGBoost: Actual vs Predicted Crop Yield")
ax.legend()
fig.tight_layout()
fig.savefig("plot_actual_vs_predicted.png", dpi=150)
print("Saved plot_actual_vs_predicted.png")

# --- Plot 2: Feature importance ---
importances = get_feature_importance(pipeline, "model").head(10)
fig, ax = plt.subplots(figsize=(7, 5))
importances.sort_values().plot(kind="barh", ax=ax, color="#558B2F")
ax.set_xlabel("Importance")
ax.set_title("Top 10 Feature Importances (XGBoost)")
fig.tight_layout()
fig.savefig("plot_feature_importance.png", dpi=150)
print("Saved plot_feature_importance.png")
