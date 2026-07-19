"""Class-imbalance handling via SMOTE."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)
