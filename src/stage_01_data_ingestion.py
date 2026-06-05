import os
import pandas as pd
from pathlib import Path

def data_ingestion():
    print("\n>>> Stage 01: Data Ingestion Started <<<")
    
    # Path configuration
    RAW_MATRIX = Path("data/raw/GSE5281_series_matrix.txt")
    SAMPLE_METADATA = Path("data/raw/GSE5281_sample_characteristics.xls")
    
    # Validation check
    if not (RAW_MATRIX.exists() and SAMPLE_METADATA.exists()):
        raise FileNotFoundError("Raw datasets are missing from data/raw/! Check DVC tracking.")
        
    # Output path setup
    os.makedirs("data/processed", exist_ok=True)
    
    print("✓ Raw files verified successfully.")
    print(">>> Stage 01: Data Ingestion Completed Successfully <<<\n")

if __name__ == "__main__":
    data_ingestion()