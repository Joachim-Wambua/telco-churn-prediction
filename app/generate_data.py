"""
app/generate_data.py — Pre-generate app data files.

Creates:
  data/predictions.csv  — test-set predictions from the production model
  data/model_results.json — structured model comparison results

Run once before launching the Streamlit app:
  python app/generate_data.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEATURES_PATH   = ROOT / "data" / "features.csv"
MODEL_PATH      = ROOT / "models" / "production_model.pkl"
PREDICTIONS_OUT = ROOT / "data" / "predictions.csv"
RESULTS_OUT     = ROOT / "data" / "model_results.json"

TARGET_COL   = "Churn"
POS_LABEL    = "Yes"
RANDOM_STATE = 42
TEST_SIZE    = 0.2


def generate_predictions() -> None:
    print(f"Loading {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    customer_ids = df["customerID"].values
    df = df.drop(columns=["customerID"])

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == POS_LABEL).astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    preds = X_test.copy()
    preds.insert(0, "customerID", customer_ids[X_test.index])
    preds["actual"]      = y_test.values
    preds["predicted"]   = y_pred
    preds["probability"] = y_prob.round(4)

    preds.to_csv(PREDICTIONS_OUT, index=False)
    n_churners = y_test.sum()
    caught = ((y_pred == 1) & (y_test == 1)).sum()
    print(f"  Saved {len(preds):,} rows -> {PREDICTIONS_OUT.name}")
    print(f"  Test churners: {n_churners} | caught: {caught} ({caught/n_churners:.1%} recall)")


def generate_model_results() -> None:
    results = {
        "models": [
            {
                "name": "Logistic Regression",
                "label": "Baseline",
                "cv_mean": None,
                "cv_std": None,
                "test_accuracy": 0.7989,
                "test_precision": 0.6417,
                "test_recall": 0.5508,
                "test_f1": 0.5928,
                "test_roc_auc": 0.8387,
                "train_time_s": 0.23,
                "winner": False,
            },
            {
                "name": "Random Forest",
                "label": "Candidate",
                "cv_mean": 0.8302,
                "cv_std": 0.0065,
                "test_accuracy": 0.7846,
                "test_precision": 0.6228,
                "test_recall": 0.4813,
                "test_f1": 0.5430,
                "test_roc_auc": 0.8140,
                "train_time_s": 14.81,
                "winner": False,
            },
            {
                "name": "Gradient Boosting",
                "label": "Candidate",
                "cv_mean": 0.8402,
                "cv_std": 0.0046,
                "test_accuracy": 0.7846,
                "test_precision": 0.6073,
                "test_recall": 0.5374,
                "test_f1": 0.5702,
                "test_roc_auc": 0.8309,
                "train_time_s": 20.18,
                "winner": False,
            },
            {
                "name": "XGBoost (Tuned)",
                "label": "Winner",
                "cv_mean": 0.8513,
                "cv_std": 0.0048,
                "test_accuracy": 0.7342,
                "test_precision": 0.5000,
                "test_recall": 0.8075,
                "test_f1": 0.6176,
                "test_roc_auc": 0.8403,
                "train_time_s": 1.05,
                "winner": True,
            },
        ]
    }

    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"  Saved -> {RESULTS_OUT.name}")


if __name__ == "__main__":
    print("=== Generating app data files ===")
    generate_predictions()
    generate_model_results()
    print("Done.")
