"""Score every candidate pair with both models and rank pairs by uncertainty."""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import DB_PATH, FEATURES, MATCHES_PATH, MODEL_DIR, PREDICTIONS_PATH
from src.review_manager import set_uncertainty


def load_models():
    feature_file = MODEL_DIR / "feature_list.json"
    if not feature_file.exists():
        raise FileNotFoundError("No trained models found. Run src/train_model.py first.")
    features = json.loads(feature_file.read_text())
    return (joblib.load(MODEL_DIR / "logistic_regression.joblib"),
            joblib.load(MODEL_DIR / "random_forest.joblib"), features)


def predict(matches):
    logistic, forest, features = load_models()
    X = pd.DataFrame({f: pd.to_numeric(matches.get(f, 0), errors="coerce")
                      for f in features}).fillna(0)

    out = matches[["record_id_A", "record_id_B", "company_A", "company_B",
                   "match_score", "match_category"]].copy()
    out["logistic_prediction"] = logistic.predict(X)
    out["random_forest_prediction"] = forest.predict(X)
    out["logistic_match_probability"] = logistic.predict_proba(X)[:, 1].round(4)
    out["random_forest_match_probability"] = forest.predict_proba(X)[:, 1].round(4)
    out["ensemble_probability"] = out[["logistic_match_probability",
                                       "random_forest_match_probability"]].mean(axis=1).round(4)
    out["predicted_label"] = (out["ensemble_probability"] >= 0.5).astype(int)
    out["model_agreement"] = out["logistic_prediction"] == out["random_forest_prediction"]
    # 1.0 = most uncertain (probability 0.5), 0.0 = fully confident
    out["uncertainty"] = (1 - np.abs(2 * out["ensemble_probability"] - 1)).round(4)
    return out


def generate_predictions():
    matches = pd.read_csv(MATCHES_PATH)
    print("Total candidate pairs:", f"{len(matches):,}")
    out = predict(matches)
    out.to_csv(PREDICTIONS_PATH, index=False)

    if DB_PATH.exists():
        set_uncertainty(out, DB_PATH)
        print("Review queue updated: most uncertain pairs will be shown first.")

    print("\n========== ML PREDICTION REPORT ==========")
    print("Predicted matches (ensemble):", int(out["predicted_label"].sum()))
    print("Model agreement:", round(out["model_agreement"].mean() * 100, 2), "%")
    print("Saved:", PREDICTIONS_PATH)
    print("NOTE: predictions are suggestions, not verified decisions.")
    return out


if __name__ == "__main__":
    generate_predictions()
