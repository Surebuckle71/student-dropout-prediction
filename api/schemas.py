"""Request/response models for the prediction API.

The real UCI dataset has 35 columns; rather than hardcode all of them (and risk drifting
from whatever's actually downloaded), `features` accepts an open dict. Unknown keys are
ignored and missing ones default to 0 at inference time -- see DropoutPredictor.predict.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    features: dict[str, float | int | str] = Field(
        ...,
        description="Raw student feature values, matching the (cleaned) dataset column names.",
        examples=[
            {
                "age_at_enrollment": 20,
                "gender": 1,
                "scholarship_holder": 0,
                "tuition_fees_up_to_date": 1,
                "debtor": 0,
                "curricular_units_1st_sem_(grade)": 12.5,
                "curricular_units_2nd_sem_(grade)": 11.0,
            }
        ],
    )


class PredictionResponse(BaseModel):
    dropout_probability: float
    at_risk: bool
    model_name: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
