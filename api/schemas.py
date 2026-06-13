from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class ModelInfoResponse(BaseModel):
    input_dim: int
    latent_dim: int
    feature_count: int
    class_labels: dict[str, int]
    training_epochs: int


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    test_samples: int


class SampleInfo(BaseModel):
    sample_id: str
    target: str
    organ_region: str


class SampleListResponse(BaseModel):
    count: int
    samples: list[SampleInfo]


class PredictionResult(BaseModel):
    label: str
    label_id: int
    probability_ad: float
    probability_control: float


class SamplePredictRequest(BaseModel):
    sample_id: str = Field(..., description="GEO sample accession, e.g. GSM119615")


class FeaturePredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=1,
        description="Scaled feature vector matching training feature order",
    )


class ExpressionPredictRequest(BaseModel):
    expression: dict[str, float] = Field(
        ...,
        description="Raw probe expression values keyed by probe ID",
    )


class BatchSamplePredictRequest(BaseModel):
    sample_ids: list[str] = Field(..., min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResult]
    sample_ids: list[str]
