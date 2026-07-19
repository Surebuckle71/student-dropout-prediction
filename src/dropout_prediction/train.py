"""End-to-end training pipeline: load -> clean -> split -> encode -> resample -> train -> select best.

Usage:
    python -m dropout_prediction.train --data dataset.csv --out models/best_model.joblib

`dataset.csv` is not included in this repo -- download it from the UCI "Predict Students'
Dropout and Academic Success" page and pass its path via --data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import binarize_target, load_and_clean, train_val_test_split
from .evaluate import compute_metrics
from .features import drop_collinear, encode_categoricals, scale_numeric
from .models import get_candidate_models
from .resampling import apply_smote


def _train_and_score(models: dict, X_train, y_train, X_val, y_val, strategy: str) -> pd.DataFrame:
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        rows.append({"model": name, "strategy": strategy, **metrics})
        fitted[(strategy, name)] = model
    return pd.DataFrame(rows), fitted


def run_training(data_path: str, out_path: str = "models/best_model.joblib", results_dir: str = "results"):
    df = load_and_clean(data_path)
    df = binarize_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    X_train_enc, X_val_enc, X_test_enc = encode_categoricals(X_train, X_val, X_test)
    X_train_enc, X_val_enc, X_test_enc = drop_collinear(X_train_enc, X_val_enc, X_test_enc)
    X_train_enc, X_val_enc, X_test_enc, scaler = scale_numeric(X_train_enc, X_val_enc, X_test_enc)
    feature_columns = X_train_enc.columns.tolist()

    # Strategy 1: class-weighted models on the original (imbalanced) training split.
    weighted_models = get_candidate_models(class_weight="balanced")
    weighted_results, weighted_fitted = _train_and_score(
        weighted_models, X_train_enc, y_train, X_val_enc, y_val, "weighted"
    )

    # Strategy 2: unweighted models on SMOTE-balanced training data.
    X_train_smote, y_train_smote = apply_smote(X_train_enc, y_train)
    smote_models = get_candidate_models()
    smote_results, smote_fitted = _train_and_score(
        smote_models, X_train_smote, y_train_smote, X_val_enc, y_val, "smote"
    )

    results_df = pd.concat([weighted_results, smote_results], ignore_index=True)
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir_path / "model_comparison.csv", index=False)

    best_row = results_df.loc[results_df["f1"].idxmax()]
    best_model = {**weighted_fitted, **smote_fitted}[(best_row["strategy"], best_row["model"])]

    y_test_pred = best_model.predict(X_test_enc)
    y_test_proba = best_model.predict_proba(X_test_enc)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "model_name": best_row["model"],
            "strategy": best_row["strategy"],
        },
        out_path,
    )
    with open(results_dir_path / "best_model_test_metrics.json", "w") as f:
        json.dump({"model": best_row["model"], "strategy": best_row["strategy"], **test_metrics}, f, indent=2)

    return results_df, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and select the best dropout-risk model.")
    parser.add_argument("--data", required=True, help="Path to dataset.csv")
    parser.add_argument("--out", default="models/best_model.joblib")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_df, test_metrics = run_training(args.data, args.out, args.results_dir)
    print(results_df.sort_values("f1", ascending=False).to_string(index=False))
    print("\nBest model test-set metrics:", test_metrics)


if __name__ == "__main__":
    main()
