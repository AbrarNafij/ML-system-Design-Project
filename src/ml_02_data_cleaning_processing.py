from __future__ import annotations

import joblib
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ml_common import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR, ensure_project_dirs, log2_transform


def run_differential_expression(log_expression: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    region_mask = metadata["organ_region"].str.contains("Entorhinal Cortex", case=False, na=False)
    ec_metadata = metadata[region_mask]
    control_ids = ec_metadata.loc[ec_metadata["target"] == "Control", "sample_id"]
    ad_ids = ec_metadata.loc[ec_metadata["target"] == "AD", "sample_id"]

    control_data = log_expression[log_expression.columns.intersection(control_ids)]
    ad_data = log_expression[log_expression.columns.intersection(ad_ids)]
    if control_data.empty or ad_data.empty:
        raise ValueError("Could not build Entorhinal Cortex AD/Control groups for differential analysis.")

    _, p_values = stats.ttest_ind(ad_data, control_data, axis=1, nan_policy="omit", equal_var=False)
    results = pd.DataFrame(
        {
            "Probe_ID": log_expression.index,
            "P_Value": p_values,
            "Log2FC": ad_data.mean(axis=1).values - control_data.mean(axis=1).values,
        }
    ).dropna()
    results["minus_log10_p"] = -results["P_Value"].clip(lower=1e-300).map(lambda value: __import__("math").log10(value))
    return results.sort_values("P_Value")


def clean_and_process(top_n_genes: int = 5000) -> None:
    ensure_project_dirs()
    print(">>> ML Stage 02: cleaning and processing started")

    expression_path = INTERIM_DIR / "expression_matrix.csv"
    metadata_path = INTERIM_DIR / "sample_metadata.csv"
    if not expression_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run ml_01_data_integration.py before processing.")

    expression = pd.read_csv(expression_path, index_col=0)
    metadata = pd.read_csv(metadata_path)

    log_expression = log2_transform(expression)
    log_expression.to_csv(PROCESSED_DIR / "expression_log2.csv")

    differential_results = run_differential_expression(log_expression, metadata)
    differential_results.to_csv(PROCESSED_DIR / "differential_expression_results.csv", index=False)

    sample_matrix = log_expression[metadata["sample_id"].tolist()].T
    gene_variance = sample_matrix.var(axis=0).sort_values(ascending=False)
    selected_genes = gene_variance.head(top_n_genes).index.tolist()
    selected_variance = gene_variance.loc[selected_genes].reset_index()
    selected_variance.columns = ["Probe_ID", "Variance"]
    selected_variance.to_csv(PROCESSED_DIR / "selected_genes.csv", index=False)

    filtered = sample_matrix[selected_genes]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filtered)
    features = pd.DataFrame(scaled, index=filtered.index, columns=selected_genes)
    features.index.name = "sample_id"
    features.to_csv(PROCESSED_DIR / "ml_features.csv")

    labels = metadata[["sample_id", "target", "target_label", "organ_region", "disease_state"]].copy()
    labels.to_csv(PROCESSED_DIR / "ml_labels.csv", index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "selected_genes": selected_genes,
            "top_n_genes": top_n_genes,
            "feature_order": selected_genes,
        },
        MODELS_DIR / "preprocessor.joblib",
    )

    print(f"Processed feature matrix saved: {features.shape}")
    print(f"Differential expression results saved: {len(differential_results)} probes")
    print(">>> ML Stage 02 complete")


if __name__ == "__main__":
    clean_and_process()
