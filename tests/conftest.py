"""Synthetic fixtures matching the UCI dataset's schema.

dataset.csv isn't checked into this repo (see README) so the test suite can't touch the
real data -- these fixtures generate a small, schema-compatible synthetic dataset instead,
enough to exercise every stage of the pipeline without needing the real download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "Marital status": rng.integers(1, 6, n),
            "Application mode": rng.integers(1, 18, n),
            "Course": rng.integers(1, 17, n),
            "Nacionality": rng.integers(1, 5, n),
            "Gender": rng.integers(0, 2, n),
            "Scholarship holder": rng.integers(0, 2, n),
            "Tuition fees up to date": rng.integers(0, 2, n),
            "Debtor": rng.integers(0, 2, n),
            "Age at enrollment": rng.integers(17, 45, n),
            "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
            "Curricular units 2nd sem (grade)": rng.uniform(0, 20, n),
            "GDP": rng.uniform(-4, 3, n),
            "Target": rng.choice(["Dropout", "Graduate", "Enrolled"], size=n, p=[0.35, 0.5, 0.15]),
        }
    )


@pytest.fixture
def cleaned_df(synthetic_raw_df: pd.DataFrame) -> pd.DataFrame:
    from dropout_prediction.data import binarize_target

    synthetic_raw_df.columns = (
        synthetic_raw_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("'", "")
    )
    df = synthetic_raw_df.drop_duplicates()
    return binarize_target(df)


@pytest.fixture
def trained_artifact(tmp_path, cleaned_df: pd.DataFrame):
    """Run the real training pipeline on synthetic data and return the saved artifact path."""
    from dropout_prediction.data import train_val_test_split
    from dropout_prediction.features import drop_collinear, encode_categoricals, scale_numeric
    from dropout_prediction.models import get_candidate_models

    import joblib

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(cleaned_df)
    X_train, X_val, X_test = encode_categoricals(X_train, X_val, X_test)
    X_train, X_val, X_test = drop_collinear(X_train, X_val, X_test)
    X_train, X_val, X_test, scaler = scale_numeric(X_train, X_val, X_test)

    model = get_candidate_models(class_weight="balanced")["logistic_regression"]
    model.fit(X_train, y_train)

    artifact_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": X_train.columns.tolist(),
            "model_name": "logistic_regression",
            "strategy": "weighted",
        },
        artifact_path,
    )
    return artifact_path
