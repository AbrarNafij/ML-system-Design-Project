from __future__ import annotations

from ml_common import (
    INTERIM_DIR,
    RAW_MATRIX,
    align_expression_to_metadata,
    ensure_project_dirs,
    load_expression_matrix,
    parse_geo_sample_metadata,
)


def integrate_data() -> None:
    ensure_project_dirs()
    print(">>> ML Stage 01: CSV data integration started")

    if not RAW_MATRIX.exists():
        raise FileNotFoundError(f"Raw GEO matrix not found: {RAW_MATRIX}")

    expression = load_expression_matrix(RAW_MATRIX)
    metadata = parse_geo_sample_metadata(RAW_MATRIX)
    expression, metadata = align_expression_to_metadata(expression, metadata)

    expression_out = INTERIM_DIR / "expression_matrix.csv"
    metadata_out = INTERIM_DIR / "sample_metadata.csv"
    expression.to_csv(expression_out)
    metadata.to_csv(metadata_out, index=False)

    print(f"Expression matrix saved: {expression_out} {expression.shape}")
    print(f"Sample metadata saved: {metadata_out} {metadata.shape}")
    print("Target breakdown:")
    print(metadata["target"].value_counts().to_string())
    print(">>> ML Stage 01 complete")


if __name__ == "__main__":
    integrate_data()
