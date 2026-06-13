from __future__ import annotations

from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(_: FastAPI):
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
