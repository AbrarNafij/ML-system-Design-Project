from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA

from ml_03_model_training import ContrastiveEncoder
from ml_common import MODELS_DIR, PROCESSED_DIR, ensure_project_dirs


FIGURES_DIR = PROCESSED_DIR / "figures"


def save_volcano_plot() -> None:
    results = pd.read_csv(PROCESSED_DIR / "differential_expression_results.csv")
    results["minus_log10_p"] = -np.log10(results["P_Value"].clip(lower=1e-300))
    significant = results[(results["P_Value"] < 0.05) & (results["Log2FC"].abs() > 1)]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results, x="Log2FC", y="minus_log10_p", color="grey", alpha=0.3, edgecolor=None)
    sns.scatterplot(data=significant, x="Log2FC", y="minus_log10_p", color="red", alpha=0.8, edgecolor=None)
    plt.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
    plt.axvline(1, color="blue", linestyle="--", linewidth=1)
    plt.axvline(-1, color="blue", linestyle="--", linewidth=1)
    plt.title("Volcano Plot: Alzheimer's vs Control (Entorhinal Cortex)")
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-log10(P-Value)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "volcano_plot.png", dpi=300)
    plt.close()


def save_top_probe_heatmap(top_n: int = 20) -> None:
    results = pd.read_csv(PROCESSED_DIR / "differential_expression_results.csv")
    expression = pd.read_csv(PROCESSED_DIR / "expression_log2.csv", index_col=0)
    labels = pd.read_csv(PROCESSED_DIR / "ml_labels.csv")

    top_probes = results.sort_values("P_Value").head(top_n)["Probe_ID"].tolist()
    sorted_labels = labels.sort_values(["organ_region", "target", "sample_id"])
    sample_ids = [sample_id for sample_id in sorted_labels["sample_id"] if sample_id in expression.columns]
    heatmap_data = expression.loc[top_probes, sample_ids]

    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap="viridis", xticklabels=False)
    plt.title(f"Expression of Top {top_n} Significant Probes")
    plt.xlabel("Samples grouped by region and target")
    plt.ylabel("Probe ID")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_probe_heatmap.png", dpi=300)
    plt.close()


def save_latent_pca_plot() -> None:
    features = pd.read_csv(PROCESSED_DIR / "ml_features.csv", index_col=0)
    labels = pd.read_csv(PROCESSED_DIR / "ml_labels.csv").set_index("sample_id").loc[features.index]
    encoder_artifact = joblib.load(MODELS_DIR / "contrastive_encoder.joblib")

    encoder = ContrastiveEncoder(
        input_dim=encoder_artifact["input_dim"],
        latent_dim=encoder_artifact["latent_dim"],
    )
    encoder.load_state_dict(encoder_artifact["state_dict"])
    encoder.eval()

    with torch.no_grad():
        latents = encoder(torch.tensor(features.values.astype(np.float32), dtype=torch.float32)).numpy()

    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(latents)
    plot_df = pd.DataFrame(pca_results, columns=["PC1", "PC2"], index=features.index)
    plot_df["Target"] = labels["target"].values
    plot_df["Region"] = labels["organ_region"].values

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Target", style="Region", s=90)
    plt.title("Latent Space PCA: Contrastive Features")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "latent_pca.png", dpi=300)
    plt.close()


def save_confusion_matrix_plot() -> None:
    confusion = pd.read_csv(PROCESSED_DIR / "confusion_matrix.csv", index_col=0)

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix: Contrastive Encoder + Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def generate_visualizations() -> None:
    ensure_project_dirs()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(">>> ML Stage 05: visualization started")

    save_volcano_plot()
    save_top_probe_heatmap()
    save_latent_pca_plot()
    save_confusion_matrix_plot()

    print(f"Saved visualizations to {FIGURES_DIR}")
    print(">>> ML Stage 05 complete")


if __name__ == "__main__":
    generate_visualizations()
