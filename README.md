# ML-system-Design-Project

## Overview

This repository contains a complete DVC-based ML pipeline for Alzheimer’s classification using the GSE5281 gene expression dataset, plus a FastAPI inference service for model predictions.

## Pipeline

- `dvc repro` builds the pipeline stages:
  - `ml_data_integration`
  - `ml_data_cleaning_processing`
  - `ml_model_training`
  - `ml_model_evaluation`
  - `ml_visualization`
- `scripts/run_pipeline.sh` runs the full DVC pipeline and then starts the FastAPI server.

## FastAPI Service

The FastAPI application is defined in `api/main.py` and exposes the following endpoints:

- `GET /health` - service health and model load status
- `GET /model/info` - loaded model metadata
- `GET /metrics` - evaluation metrics from `data/processed/model_metrics.json`
- `GET /samples` - sample metadata available for prediction
- `POST /predict/sample` - predict a sample by `sample_id`
- `POST /predict/samples` - batch predict multiple sample IDs
- `POST /predict/features` - predict from a preprocessed feature vector
- `POST /predict/expression` - predict from raw expression values keyed by probe IDs

The interactive OpenAPI docs are available at `http://127.0.0.1:8000/docs` once the server is running.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./scripts/run_pipeline.sh
```

If you only want to start the API after the artifacts are already built:

```bash
./scripts/run_api.sh
```

## Notes

- Model artifacts are stored under `models/`.
- Processed data and evaluation outputs are stored under `data/processed/`.
- The API loader will raise a clear error if required artifacts are missing.
