"""
yield_prediction_pipeline.py

Precision Agriculture - Crop Yield Prediction
AIBUY3A Business Analysis 3.2 Project

Pipeline stages:
1. Load data
2. Clean data (handle missing values, outliers)
3. Feature engineering
4. Train/test split + preprocessing
5. Train Random Forest and XGBoost regressors
6. Evaluate (RMSE, R^2, 5-fold cross-validation)
7. Feature importance (explainability for the "business objectives" section)
8. Save the best model
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_PATH = "sa_crop_yield_dataset.csv"
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Report missing values before handling (useful for your documentation)
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts):
        print("Missing values found:\n", missing_counts)

    # Remove statistically implausible outliers using the IQR method on the
    # target variable - this catches the injected data-entry-error records
    q1, q3 = df["yield_tons_per_ha"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[(df["yield_tons_per_ha"] >= lower) & (df["yield_tons_per_ha"] <= upper)]
    print(f"Removed {before - len(df)} outlier rows using IQR method "
          f"(valid range: {lower:.2f} - {upper:.2f} tons/ha)")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total water available to the crop - a single engineered feature that
    # combines rainfall and irrigation, often more predictive than either alone
    df["total_water_mm"] = df["season_rainfall_mm"] + df["irrigation_mm"]

    # Fertilizer applied per hectare relative to farm size gives a sense of
    # input intensity independent of farm scale
    df["fertilizer_intensity"] = df["fertilizer_kg_ha"] / np.log1p(df["farm_size_ha"])

    # Simple heat-stress flag: crops above 26C start losing yield in this model
    df["heat_stress_flag"] = (df["avg_temp_c"] > 26).astype(int)

    return df


# ---------------------------------------------------------------------------
# 4. Build preprocessing + train/test split
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "season_rainfall_mm", "avg_temp_c", "soil_moisture_pct", "irrigation_mm",
    "fertilizer_kg_ha", "pest_pressure_index", "farm_size_ha",
    "total_water_mm", "fertilizer_intensity", "heat_stress_flag",
]
CATEGORICAL_FEATURES = ["province", "crop_type", "soil_type"]
TARGET = "yield_tons_per_ha"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# ---------------------------------------------------------------------------
# 5-6. Train + evaluate models
# ---------------------------------------------------------------------------
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")

    print(f"\n--- {name} ---")
    print(f"Test RMSE: {rmse:.3f} tons/ha")
    print(f"Test R^2:  {r2:.3f}")
    print(f"5-fold CV R^2: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    return {"name": name, "pipeline": pipeline, "rmse": rmse, "r2": r2,
             "cv_r2_mean": cv_scores.mean()}


def get_feature_importance(pipeline, model_step_name):
    """Extract and print feature importance from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps[model_step_name]

    cat_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    all_feature_names = NUMERIC_FEATURES + list(cat_names)

    importances = pd.Series(model.feature_importances_, index=all_feature_names)
    importances = importances.sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(importances.head(10).to_string())
    return importances


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        )),
    ])

    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_SEED, n_jobs=-1
        )),
    ])

    results = []
    results.append(evaluate_model("Random Forest", rf_pipeline, X_train, X_test, y_train, y_test))
    results.append(evaluate_model("XGBoost", xgb_pipeline, X_train, X_test, y_train, y_test))

    best = max(results, key=lambda r: r["r2"])
    print(f"\n=== Best model: {best['name']} (Test R^2 = {best['r2']:.3f}) ===")

    get_feature_importance(best["pipeline"], "model")

    model_path = "best_yield_model.joblib"
    joblib.dump(best["pipeline"], model_path)
    print(f"\nSaved best model pipeline -> {model_path}")


if __name__ == "__main__":
    main()
