"""Train Logistic Regression + Random Forest on human-reviewed pairs.

Unlike a plain fit, this reports *cross-validated* precision / recall / F1 /
ROC-AUC and compares the models with a simple rule baseline.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES, MATCHES_PATH, METRICS_PATH, MIN_RECOMMENDED_LABELS, MODEL_DIR,
    REPORT_DIR, TRAINING_PATH,
)

BASELINE_THRESHOLD = 0.90   # rule baseline: predict match when score >= 0.90


def load_training_data(training_path=TRAINING_PATH, matches_path=MATCHES_PATH):
    labels = pd.read_csv(training_path)
    matches = pd.read_csv(matches_path)
    keep = ["record_id_A", "record_id_B", "match_score"] + [
        f for f in FEATURES if f in matches.columns]
    data = labels[["record_id_A", "record_id_B", "label"]].merge(
        matches[keep], on=["record_id_A", "record_id_B"], how="inner")
    dropped = len(labels) - len(data)
    if data.empty:
        raise ValueError("None of the reviewed pairs exist in potential_matches.csv. "
                         "Re-run duplicate_detection.py first.")
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data.get(feature, 0), errors="coerce").fillna(0)
    data["label"] = data["label"].astype(int)
    return data, dropped


def build_models():
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000,
                                          random_state=42)),
    ])
    forest = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2,
                                    class_weight="balanced_subsample", random_state=42)
    return {"logistic_regression": logistic, "random_forest": forest}


def _metrics(y, pred, proba):
    out = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }
    try:
        out["roc_auc"] = roc_auc_score(y, proba)
    except ValueError:
        out["roc_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def cross_validate(models, X, y, repeats=5):
    """Repeated stratified CV. Returns {model: {metric: {mean, std}}}."""
    minority = int(min((y == 0).sum(), (y == 1).sum()))
    folds = min(5, minority)
    if folds < 2:
        return None, folds

    runs = {name: [] for name in list(models) + ["rule_baseline"]}
    score = X["match_score"].to_numpy()
    for seed in range(repeats):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for name, model in models.items():
            proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            runs[name].append(_metrics(y, (proba >= 0.5).astype(int), proba))
    baseline = _metrics(y, (score >= BASELINE_THRESHOLD).astype(int), score)
    runs["rule_baseline"] = [baseline]

    summary = {}
    for name, results in runs.items():
        summary[name] = {}
        for metric in ("precision", "recall", "f1", "accuracy", "roc_auc"):
            values = np.array([r[metric] for r in results], dtype=float)
            summary[name][metric] = {"mean": round(float(np.nanmean(values)), 3),
                                     "std": round(float(np.nanstd(values)), 3)}
        summary[name]["confusion_last_run"] = results[-1]["confusion"]
    return summary, folds


def train_models():
    print("Loading human-reviewed training data...")
    data, dropped = load_training_data()
    y = data["label"]
    X = data[FEATURES + ["match_score"]]

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"Examples: {len(data)} (approved {n_pos}, rejected {n_neg}); "
          f"{dropped} labelled pairs no longer in the candidate set")
    if y.nunique() < 2:
        print("Both approved and rejected examples are required.")
        return None

    warnings = []
    if len(data) < MIN_RECOMMENDED_LABELS:
        warnings.append(
            f"Only {len(data)} labelled pairs (recommended: {MIN_RECOMMENDED_LABELS}+). "
            "Metrics below are noisy - review more pairs (active learning queue).")
    if dropped:
        warnings.append(f"{dropped} labelled pairs were not found in potential_matches.csv.")

    models = build_models()
    cv_summary, folds = cross_validate(models, X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(X[FEATURES], y)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    (MODEL_DIR / "feature_list.json").write_text(json.dumps(FEATURES, indent=2))

    importance = sorted(zip(FEATURES, models["random_forest"].feature_importances_),
                        key=lambda t: -t[1])
    report = {
        "n_labels": len(data), "n_approved": n_pos, "n_rejected": n_neg,
        "dropped_labels": dropped, "cv_folds": folds,
        "cross_validation": cv_summary, "warnings": warnings,
        "feature_importance": {k: round(float(v), 4) for k, v in importance},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print("\n========== MODEL TRAINING REPORT ==========")
    if cv_summary:
        print(f"{folds}-fold stratified CV (repeated):")
        for name, res in cv_summary.items():
            print(f"  {name:20s} P={res['precision']['mean']:.2f} "
                  f"R={res['recall']['mean']:.2f} F1={res['f1']['mean']:.2f} "
                  f"AUC={res['roc_auc']['mean']:.2f}")
    else:
        print("Not enough examples per class for cross-validation.")
    print("\nTop features:", ", ".join(f"{k} ({v:.2f})" for k, v in importance[:5]))
    for w in warnings:
        print("WARNING:", w)
    print("Saved:", METRICS_PATH)
    return report


if __name__ == "__main__":
    train_models()
