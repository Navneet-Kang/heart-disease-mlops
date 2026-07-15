"""FastAPI application for heart-disease prediction."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the probability of heart disease from patient health data.",
    version="1.0.0",
)

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "heart_api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

model = None
model_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    else:
        model_error = f"Model file not found: {MODEL_PATH}"
        logger.warning(model_error)
except Exception as exc:
    model_error = str(exc)
    logger.exception("Failed to load model")


class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float | None = Field(default=None, ge=0, le=3)
    thal: float | None = Field(default=None)


@app.middleware("http")
async def request_monitoring(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint=request.url.path,
        ).observe(elapsed)

        logger.info(
            "%s %s status=%s latency=%.4fs",
            request.method,
            request.url.path,
            status_code,
            elapsed,
        )


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "error": model_error,
    }


@app.post("/predict")
def predict(patient: PatientInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is unavailable. Run `python -m src.train` "
                "and confirm models/heart_disease_pipeline.joblib exists."
            ),
        )

    frame = pd.DataFrame([patient.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "risk": "high" if prediction == 1 else "low",
        "confidence": round(probability, 4),
        "threshold": 0.5,
    }
