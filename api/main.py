from __future__ import annotations

from contextlib import asynccontextmanager
import os

import numpy as np
from fastapi import FastAPI, HTTPException

from api.predictor import ModelPredictor
from api.schemas import (
    BatchPredictionResponse,
    BatchSamplePredictRequest,
    ExpressionPredictRequest,
    FeaturePredictRequest,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResult,
    SampleListResponse,
    SamplePredictRequest,
)

predictor = ModelPredictor()

# Enable a development mode that skips loading large ML artifacts at startup.
# Set the environment variable `API_DEV_MODE=1` to enable dev mode.
DEV_MODE = os.getenv("API_DEV_MODE", "0") == "1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not DEV_MODE:
        predictor.load()
    yield


app = FastAPI(
    title="GSE5281 Alzheimer's Classification API",
    description="Inference service for the contrastive encoder + Random Forest pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=predictor.is_loaded)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**predictor.get_model_info())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics", response_model=MetricsResponse, tags=["model"])
def metrics() -> MetricsResponse:
    try:
        return MetricsResponse(**predictor.get_metrics())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/model/load", tags=["model"])
def load_models() -> dict:
    """Trigger model loading at runtime (useful in dev mode)."""
    try:
        predictor.load()
        return {"loaded": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/samples", response_model=SampleListResponse, tags=["data"])
def list_samples() -> SampleListResponse:
    samples = predictor.list_samples()
    return SampleListResponse(count=len(samples), samples=samples)


@app.post("/predict/sample", response_model=PredictionResult, tags=["predict"])
def predict_sample(request: SamplePredictRequest) -> PredictionResult:
    try:
        return PredictionResult(**predictor.predict_sample(request.sample_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/samples", response_model=BatchPredictionResponse, tags=["predict"])
def predict_samples(request: BatchSamplePredictRequest) -> BatchPredictionResponse:
    try:
        predictions = [PredictionResult(**item) for item in predictor.predict_batch_samples(request.sample_ids)]
        return BatchPredictionResponse(predictions=predictions, sample_ids=request.sample_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/features", response_model=PredictionResult, tags=["predict"])
def predict_features(request: FeaturePredictRequest) -> PredictionResult:
    try:
        return PredictionResult(**predictor.predict_from_features(np.array(request.features)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/expression", response_model=PredictionResult, tags=["predict"])
def predict_expression(request: ExpressionPredictRequest) -> PredictionResult:
    try:
        return PredictionResult(**predictor.predict_from_expression(request.expression))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
