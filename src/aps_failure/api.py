"""FastAPI inference service for Scania APS failure prediction."""

from functools import lru_cache
import os
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from aps_failure.model_io import DEFAULT_MODEL_PATH, load_model_bundle, make_inference_frame


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="APS sensor feature values keyed by their original column names.",
    )


class PredictionResponse(BaseModel):
    failure_prediction: int
    failure_probability: float


@lru_cache
def get_model_bundle() -> dict:
    model_path = os.getenv("APS_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    return load_model_bundle(model_path)


app = FastAPI(
    title="Scania APS Failure Prediction API",
    version="0.1.0",
    description="Inference API for an XGBoost model trained on Scania APS sensor data.",
)


@app.get("/")
def root():
    return {"status": "ok", "service": "scania-aps-failure-api"}


@app.get("/health")
def health_check():
    bundle = get_model_bundle()
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(bundle["feature_names"]),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    bundle = get_model_bundle()
    model = bundle["model"]
    frame = make_inference_frame(payload.features, bundle["feature_names"])
    prediction = model.predict(frame)[0]
    probability = model.predict_proba(frame)[0][1]
    return PredictionResponse(
        failure_prediction=int(prediction),
        failure_probability=float(probability),
    )
