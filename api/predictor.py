from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_03_model_training import ContrastiveEncoder  # noqa: E402
from ml_common import MODELS_DIR, PROCESSED_DIR  # noqa: E402

LABEL_NAMES = {0: "Control", 1: "AD"}


class ModelPredictor:
    def __init__(self) -> None:
        self.encoder: ContrastiveEncoder | None = None
        self.classifier = None
        self.preprocessor: dict | None = None
        self.encoder_artifact: dict | None = None
        self.feature_index: pd.Index | None = None
        self.labels_df: pd.DataFrame | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        encoder_path = MODELS_DIR / "contrastive_encoder.joblib"
        classifier_path = MODELS_DIR / "random_forest_classifier.joblib"
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        features_path = PROCESSED_DIR / "ml_features.csv"
        labels_path = PROCESSED_DIR / "ml_labels.csv"

        for path in (encoder_path, classifier_path, preprocessor_path, features_path, labels_path):
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}. Run `dvc repro` first.")

        self.encoder_artifact = joblib.load(encoder_path)
        self.classifier = joblib.load(classifier_path)
        self.preprocessor = joblib.load(preprocessor_path)

        self.encoder = ContrastiveEncoder(
            input_dim=self.encoder_artifact["input_dim"],
            latent_dim=self.encoder_artifact["latent_dim"],
        )
        self.encoder.load_state_dict(self.encoder_artifact["state_dict"])
        self.encoder.eval()

        self.feature_index = pd.read_csv(features_path, index_col=0, nrows=0).columns
        self.labels_df = pd.read_csv(labels_path)

        self._loaded = True

    def get_model_info(self) -> dict:
        if not self.encoder_artifact or not self.preprocessor:
            raise RuntimeError("Models are not loaded.")
        return {
            "input_dim": self.encoder_artifact["input_dim"],
            "latent_dim": self.encoder_artifact["latent_dim"],
            "feature_count": len(self.preprocessor["selected_genes"]),
            "class_labels": self.encoder_artifact["class_mapping"],
            "training_epochs": self.encoder_artifact["epochs"],
        }

    def list_samples(self) -> list[dict]:
        if self.labels_df is None:
            raise RuntimeError("Labels are not loaded.")
        return [
            {
                "sample_id": row["sample_id"],
                "target": row["target"],
                "organ_region": row["organ_region"],
            }
            for _, row in self.labels_df.iterrows()
        ]

    def get_metrics(self) -> dict:
        metrics_path = PROCESSED_DIR / "model_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        with metrics_path.open() as handle:
            return json.load(handle)

    def predict_from_features(self, features: np.ndarray) -> dict:
        self._ensure_loaded()
        expected = self.encoder_artifact["input_dim"]
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != expected:
            raise ValueError(f"Expected {expected} features, got {features.shape[1]}.")

        latents = self._encode(features.astype(np.float32))
        return self._classify(latents)[0]

    def predict_sample(self, sample_id: str) -> dict:
        self._ensure_loaded()
        features_path = PROCESSED_DIR / "ml_features.csv"
        features_df = pd.read_csv(features_path, index_col=0)
        if sample_id not in features_df.index:
            raise KeyError(f"Unknown sample_id: {sample_id}")

        row = features_df.loc[[sample_id]].values.astype(np.float32)
        return self.predict_from_features(row)

    def predict_batch_samples(self, sample_ids: list[str]) -> list[dict]:
        return [self.predict_sample(sample_id) for sample_id in sample_ids]

    def predict_from_expression(self, expression: dict[str, float]) -> dict:
        self._ensure_loaded()
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not loaded.")

        series = pd.Series(expression, dtype=float)
        min_value = float(series.min())
        offset = 1.0 if min_value >= 0 else abs(min_value) + 1.0
        log_expression = np.log2(series + offset)

        selected_genes = self.preprocessor["selected_genes"]
        selected = log_expression.reindex(selected_genes)
        missing = selected[selected.isna()].index.tolist()
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(f"Missing {len(missing)} required probes. Examples: {preview}{suffix}")

        scaled = self.preprocessor["scaler"].transform(selected.values.reshape(1, -1))
        return self.predict_from_features(scaled)

    def _encode(self, features: np.ndarray) -> np.ndarray:
        assert self.encoder is not None
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32)
            return self.encoder(tensor).numpy()

    def _classify(self, latents: np.ndarray) -> list[dict]:
        assert self.classifier is not None
        probabilities = self.classifier.predict_proba(latents)
        predictions = self.classifier.predict(latents)

        results = []
        for pred, probs in zip(predictions, probabilities):
            label_id = int(pred)
            prob_control = float(probs[0])
            prob_ad = float(probs[1]) if len(probs) > 1 else 1.0 - prob_control
            results.append(
                {
                    "label": LABEL_NAMES[label_id],
                    "label_id": label_id,
                    "probability_ad": prob_ad,
                    "probability_control": prob_control,
                }
            )
        return results

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Models are not loaded.")
