"""Data download, cleaning, and validation for the UCI Cleveland dataset."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
TARGET = "target"
FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
COLUMNS = FEATURES + [TARGET]
NUMERIC_FEATURES = ["age","trestbps","chol","thalach","oldpeak","ca"]
CATEGORICAL_FEATURES = ["sex","cp","fbs","restecg","exang","slope","thal"]

def project_paths(project_root="."):
    root = Path(project_root).resolve()
    return root / "data/raw/processed.cleveland.data", root / "data/processed/heart_disease.csv"

def download_dataset(destination, overwrite=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or overwrite:
        urlretrieve(DATA_URL, destination)
    return destination

def validate_dataset(data):
    missing = sorted(set(COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data.empty or len(data) < 100:
        raise ValueError("Dataset is empty or unexpectedly small.")
    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values.")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary.")

def clean_dataset(raw_path):
    data = pd.read_csv(raw_path, names=COLUMNS, na_values="?")
    for column in COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[TARGET] = (data[TARGET] > 0).astype(int)
    data = data.drop_duplicates().reset_index(drop=True)
    validate_dataset(data)
    return data

def prepare_dataset(project_root=".", force_download=False):
    raw_path, processed_path = project_paths(project_root)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    if processed_path.exists() and not force_download:
        validate_dataset(pd.read_csv(processed_path))
        return processed_path
    download_dataset(raw_path, overwrite=force_download)
    clean_dataset(raw_path).to_csv(processed_path, index=False)
    return processed_path

def load_processed_data(project_root="."):
    path = prepare_dataset(project_root)
    data = pd.read_csv(path)
    validate_dataset(data)
    return data

if __name__ == "__main__":
    path = prepare_dataset()
    data = pd.read_csv(path)
    print(f"Saved: {path}")
    print(f"Shape: {data.shape}")
    print(data[TARGET].value_counts().sort_index())
