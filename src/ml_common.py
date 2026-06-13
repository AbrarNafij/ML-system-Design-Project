from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


RAW_MATRIX = Path("data/raw/GSE5281_series_matrix.txt")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def ensure_project_dirs() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def clean_geo_value(value: object) -> str:
    return str(value).strip().strip('"').replace("\xa0", " ").strip()


def find_series_matrix_table_start(matrix_path: Path = RAW_MATRIX) -> int:
    with matrix_path.open(errors="replace") as handle:
        for index, line in enumerate(handle):
            if line.startswith("!series_matrix_table_begin"):
                return index + 1
    raise ValueError(f"Could not find !series_matrix_table_begin in {matrix_path}")


def parse_geo_sample_metadata(matrix_path: Path = RAW_MATRIX) -> pd.DataFrame:
    metadata_rows: dict[str, list[str]] = {}
    characteristics: dict[str, list[str]] = {}

    with matrix_path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break

            parts = [clean_geo_value(part) for part in line.rstrip("\n").split("\t")]
            if len(parts) <= 1:
                continue

            key, values = parts[0], parts[1:]
            if key == "!Sample_characteristics_ch1":
                parsed = [value.split(":", 1) for value in values]
                labels = [item[0].strip().lower() if len(item) == 2 else "characteristic" for item in parsed]
                vals = [item[1].strip() if len(item) == 2 else item[0].strip() for item in parsed]
                label = max(set(labels), key=labels.count).replace(" ", "_").replace("-", "_")
                characteristics[label] = vals
            elif key.startswith("!Sample_"):
                metadata_rows[key.removeprefix("!Sample_").lower()] = values

    accessions = metadata_rows.get("geo_accession")
    if not accessions:
        raise ValueError("Could not parse sample accessions from GEO matrix metadata.")

    metadata = pd.DataFrame({"sample_id": accessions})
    for key, values in metadata_rows.items():
        if len(values) == len(metadata):
            metadata[key] = values
    for key, values in characteristics.items():
        if len(values) == len(metadata):
            metadata[key] = values

    if "organ_region" not in metadata and "source_name_ch1" in metadata:
        metadata["organ_region"] = metadata["source_name_ch1"].str.replace("brain,", "", regex=False).str.strip()

    disease_source = metadata.get("disease_state", metadata.get("title", pd.Series("", index=metadata.index)))
    metadata["target"] = disease_source.apply(normalize_target)
    metadata["target_label"] = metadata["target"].map({"Control": 0, "AD": 1}).astype(int)
    return metadata


def normalize_target(value: object) -> str:
    text = str(value).lower()
    if "alzheimer" in text or re.search(r"\bad\b", text) or "affected" in text:
        return "AD"
    return "Control"


def load_expression_matrix(matrix_path: Path = RAW_MATRIX) -> pd.DataFrame:
    skiprows = find_series_matrix_table_start(matrix_path)
    matrix = pd.read_csv(matrix_path, sep="\t", skiprows=skiprows, low_memory=False)
    matrix.columns = [clean_geo_value(column) for column in matrix.columns]

    index_col = "ID_REF" if "ID_REF" in matrix.columns else matrix.columns[0]
    matrix = matrix.set_index(index_col)
    matrix.index = matrix.index.map(clean_geo_value)
    matrix = matrix[~matrix.index.str.contains("series_matrix_table_end", case=False, na=False)]
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.dropna(how="all")
    return matrix


def align_expression_to_metadata(expression: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = [sample_id for sample_id in metadata["sample_id"].tolist() if sample_id in expression.columns]
    if not sample_ids:
        raise ValueError("No metadata sample IDs match expression matrix columns.")

    aligned_metadata = metadata.set_index("sample_id").loc[sample_ids].reset_index()
    aligned_expression = expression[sample_ids]
    return aligned_expression, aligned_metadata


def log2_transform(expression: pd.DataFrame) -> pd.DataFrame:
    numeric = expression.astype(float)
    min_value = np.nanmin(numeric.values)
    offset = 1.0 if min_value >= 0 else abs(min_value) + 1.0
    return np.log2(numeric + offset)
