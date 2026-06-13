from __future__ import annotations

import argparse
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_common import MODELS_DIR, PROCESSED_DIR, ensure_project_dirs


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def supervised_contrastive_proxy_loss(latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(latents, latents)
    same_class = (labels == labels.T).float()
    different_class = 1.0 - same_class
    return (same_class * distances).mean() - (different_class * distances).mean()


def train_models(epochs: int = 150, latent_dim: int = 32, random_state: int = 42) -> None:
    ensure_project_dirs()
    set_seed(random_state)
    print(">>> ML Stage 03: model training started")

    features = pd.read_csv(PROCESSED_DIR / "ml_features.csv", index_col=0)
    labels = pd.read_csv(PROCESSED_DIR / "ml_labels.csv").set_index("sample_id").loc[features.index]
    X = features.values.astype(np.float32)
    y = labels["target_label"].values.astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=random_state, stratify=y
    )

    X_train_tensor = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train_tensor = torch.tensor(y[train_idx], dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X[test_idx], dtype=torch.float32)

    encoder = ContrastiveEncoder(input_dim=X.shape[1], latent_dim=latent_dim)
    optimizer = optim.Adam(encoder.parameters(), lr=0.0005)

    for epoch in range(epochs):
        encoder.train()
        optimizer.zero_grad()
        latents = encoder(X_train_tensor)
        loss = supervised_contrastive_proxy_loss(latents, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch + 1}/{epochs} loss={loss.item():.6f}")

    encoder.eval()
    with torch.no_grad():
        train_latents = encoder(X_train_tensor).numpy()
        test_latents = encoder(X_test_tensor).numpy()

    classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    classifier.fit(train_latents, y[train_idx])

    joblib.dump(
        {
            "state_dict": encoder.state_dict(),
            "input_dim": X.shape[1],
            "latent_dim": latent_dim,
            "epochs": epochs,
            "random_state": random_state,
            "feature_columns": features.columns.tolist(),
            "class_mapping": {"Control": 0, "AD": 1},
        },
        MODELS_DIR / "contrastive_encoder.joblib",
    )
    joblib.dump(classifier, MODELS_DIR / "random_forest_classifier.joblib")
    joblib.dump(
        {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "sample_ids": features.index.tolist(),
            "y": y,
            "train_latents": train_latents,
            "test_latents": test_latents,
        },
        MODELS_DIR / "training_split.joblib",
    )

    print(f"Saved encoder: {MODELS_DIR / 'contrastive_encoder.joblib'}")
    print(f"Saved classifier: {MODELS_DIR / 'random_forest_classifier.joblib'}")
    print(">>> ML Stage 03 complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GSE5281 contrastive encoder and RF classifier.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(epochs=args.epochs, latent_dim=args.latent_dim, random_state=args.random_state)
