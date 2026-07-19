"""Feature encoding, multicollinearity pruning, and scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def infer_categorical_columns(df: pd.DataFrame, max_cardinality: int = 30) -> list[str]:
    """Flag object columns and low-cardinality integer-coded columns (e.g. course, gender)
    as categorical. The UCI dataset encodes categoricals as integer codes, so dtype alone
    isn't enough -- verify this list against your actual columns before training on real data.
    """
    categorical = []
    for col in df.columns:
        # pandas 3.x defaults string columns to a dedicated StringDtype, not `object`,
        # so check both rather than relying on `dtype == object`.
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= max_cardinality:
            categorical.append(col)
    return categorical


def encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list[str] | None = None,
):
    """One-hot encode categoricals, aligning val/test columns to train."""
    if categorical_cols is None:
        categorical_cols = infer_categorical_columns(X_train)

    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[categorical_cols] = df[categorical_cols].astype(str)
        return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    train_enc = _encode(X_train)
    val_enc = _encode(X_val).reindex(columns=train_enc.columns, fill_value=0)
    test_enc = _encode(X_test).reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, val_enc, test_enc


def drop_collinear(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.9,
):
    """Drop one feature from any pair whose absolute correlation exceeds threshold."""
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    return X_train.drop(columns=to_drop), X_val.drop(columns=to_drop), X_test.drop(columns=to_drop)


def scale_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list[str] | None = None,
):
    """Fit a StandardScaler on train only, apply to all three splits."""
    if numeric_cols is None:
        numeric_cols = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_val, X_test, scaler
