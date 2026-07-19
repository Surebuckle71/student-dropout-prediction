"""Inference wrapper: load a trained artifact and score a single student's features."""

from __future__ import annotations

import joblib
import pandas as pd


class DropoutPredictor:
    def __init__(self, model_path: str):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_columns: list[str] = artifact["feature_columns"]
        self.model_name = artifact.get("model_name")
        self.strategy = artifact.get("strategy")

    def predict(self, features: dict) -> dict:
        row = pd.DataFrame([features])
        row = pd.get_dummies(row)
        row = row.reindex(columns=self.feature_columns, fill_value=0)
        row[self.feature_columns] = self.scaler.transform(row[self.feature_columns])

        proba = float(self.model.predict_proba(row)[0, 1])
        return {
            "dropout_probability": proba,
            "at_risk": proba >= 0.5,
            "model_name": self.model_name,
        }
