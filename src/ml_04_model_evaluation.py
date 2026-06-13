from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_03_model_training import ContrastiveEncoder
from ml_common import MODELS_DIR, PROCESSED_DIR, ensure_project_dirs


def evaluate_models() -> None:
    ensure_project_dirs()
    print(">>> ML Stage 04: model evaluation started")

    features = pd.read_csv(PROCESSED_DIR / "ml_features.csv", index_col=0)
    labels = pd.read_csv(PROCESSED_DIR / "ml_labels.csv").set_index("sample_id").loc[features.index]
    encoder_artifact = joblib.load(MODELS_DIR / "contrastive_encoder.joblib")
    classifier = joblib.load(MODELS_DIR / "random_forest_classifier.joblib")
    split = joblib.load(MODELS_DIR / "training_split.joblib")

    encoder = ContrastiveEncoder(
        input_dim=encoder_artifact["input_dim"],
        latent_dim=encoder_artifact["latent_dim"],
    )
    encoder.load_state_dict(encoder_artifact["state_dict"])
    encoder.eval()

    test_idx = split["test_idx"]
    X_test = features.values.astype(np.float32)[test_idx]
    y_test = labels["target_label"].values.astype(int)[test_idx]

    with torch.no_grad():
        test_latents = encoder(torch.tensor(X_test, dtype=torch.float32)).numpy()

    y_pred = classifier.predict(test_latents)
    y_prob = classifier.predict_proba(test_latents)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else None,
        "test_samples": int(len(y_test)),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Control", "AD"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    with (PROCESSED_DIR / "model_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame(report).T.to_csv(PROCESSED_DIR / "classification_report.csv")
    pd.DataFrame(cm, index=["actual_control", "actual_ad"], columns=["pred_control", "pred_ad"]).to_csv(
        PROCESSED_DIR / "confusion_matrix.csv"
    )

    print(json.dumps(metrics, indent=2))
    print(">>> ML Stage 04 complete")


if __name__ == "__main__":
    evaluate_models()
