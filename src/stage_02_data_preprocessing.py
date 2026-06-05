import os
import pandas as pd
from pathlib import Path

def preprocess_data():
    print("\n>>> Stage 02: Data Preprocessing Started <<<")
    
    RAW_MATRIX = Path("data/raw/GSE5281_series_matrix.txt")
    SAMPLE_METADATA = Path("data/raw/GSE5281_sample_characteristics.xls")
    
    if not RAW_MATRIX.exists():
        raise FileNotFoundError(f"Raw matrix file missing at: {RAW_MATRIX}")

    # 1. Dynamic Header Row Finder logic for GSE Matrix
    print("Finding the correct header row dynamically...")
    skip_rows = 0
    with open(RAW_MATRIX, 'r') as f:
        for idx, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skip_rows = idx + 1
                break
                
    if skip_rows == 0:
        # Default fallback options if marker is missing
        skip_rows = 32
        
    print(f"Skipping metadata header blocks. Starting parse from row: {skip_rows}")

    # 2. Load Expression Matrix avoiding Dtype and KeyError warnings
    print("Loading genomic expression matrix...")
    df_matrix = pd.read_csv(RAW_MATRIX, sep="\t", skiprows=skip_rows, low_memory=False)
    
    # Strip spaces from column headers to fix alignment hidden mismatches
    df_matrix.columns = df_matrix.columns.str.strip()
    
    # Target validation checking for 'ID_REF' column identifier
    if "ID_REF" in df_matrix.columns:
        df_matrix.set_index("ID_REF", inplace=True)
    elif df_matrix.columns[0] != "":
        # Fallback automatic structural setup if the name is slightly different
        print(f"⚠️ 'ID_REF' not found directly. Using first column as index: '{df_matrix.columns[0]}'")
        df_matrix.set_index(df_matrix.columns[0], inplace=True)
    else:
        raise KeyError("Could not determine the index/probe column from the dataset layout.")
    
    # End of file clean up if exists
    if "!series_matrix_table_end" in df_matrix.index:
        df_matrix = df_matrix.drop("!series_matrix_table_end")
        
    # 3. Load Characteristics Metadata safely
    print("Loading sample characteristics metadata...")
    if SAMPLE_METADATA.exists():
        df_meta = pd.read_excel(SAMPLE_METADATA, header=None)
        df_meta = df_meta.map(lambda s: s.strip() if isinstance(s, str) else s)
    else:
        print("⚠️ Warning: Sample metadata excel file not found. Skipping metadata loading step.")
    
    # 4. Save clean processed file configuration
    os.makedirs("data/processed", exist_ok=True)
    matrix_out_path = Path("data/processed/matrix_clean.csv")
    df_matrix.to_csv(matrix_out_path)
    
    print(f"✓ Preprocessed matrix saved successfully. Dimensions: {df_matrix.shape}")
    print(">>> Stage 02: Data Preprocessing Completed Successfully <<<\n")

if __name__ == "__main__":
    preprocess_data()