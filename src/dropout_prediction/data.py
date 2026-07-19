"""Loading, cleaning, and splitting the UCI student dropout dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw CSV and normalize column names (lowercase, underscores)."""
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("'", "")
    )
    return df.drop_duplicates()


def binarize_target(
    df: pd.DataFrame,
    target_col: str = "target",
    drop_label: str = "enrolled",
    positive_label: str = "dropout",
) -> pd.DataFrame:
    """Drop the ambiguous 'still enrolled' class and binarize dropout vs. graduate."""
    df = df[df[target_col].str.lower() != drop_label].copy()
    df["target_binary"] = (df[target_col].str.lower() == positive_label).astype(int)
    return df.drop(columns=[target_col])


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str = "target_binary",
    random_state: int = 42,
):
    """60/20/20 stratified train/val/test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
