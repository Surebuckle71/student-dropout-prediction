"""FastAPI inference service for the student dropout risk model.

Run locally:
    uvicorn api.main:app --reload

Requires a trained model artifact (see dropout_prediction.train); point at one via the
DROPOUT_MODEL_PATH env var, or drop it at the default models/best_model.joblib.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from dropout_prediction.predict import DropoutPredictor

from .schemas import HealthResponse, PredictionResponse, StudentFeatures

app = FastAPI(
    title="Student Dropout Risk API",
    description="Predicts a student's probability of dropping out from early academic and demographic signals.",
    version="0.1.0",
)

_predictor: DropoutPredictor | None = None


def _model_path() -> str:
    return os.environ.get("DROPOUT_MODEL_PATH", "models/best_model.joblib")


def get_predictor() -> DropoutPredictor:
    global _predictor
    if _predictor is None:
        path = _model_path()
        if not Path(path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"No model artifact at '{path}'. Run `python -m dropout_prediction.train --data dataset.csv` first.",
            )
        _predictor = DropoutPredictor(path)
    return _predictor


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=Path(_model_path()).exists())


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentFeatures) -> PredictionResponse:
    predictor = get_predictor()
    result = predictor.predict(payload.features)
    return PredictionResponse(**result)
