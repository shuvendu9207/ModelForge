"""
Dataset Loader
Loads datasets from CSV, JSON, and versioned storage.
"""

import os
import json
import hashlib
import shutil
import pandas as pd
from datetime import datetime


class DatasetLoader:

    VERSIONS_DIR = "data/versions"

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load a CSV or JSON file into a DataFrame."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .csv or .json")

        DatasetLoader._validate(df)
        DatasetLoader._version(path)
        print(f"[Loader] Loaded {len(df)} rows × {len(df.columns)} columns from {path}")
        return df

    @staticmethod
    def _validate(df: pd.DataFrame):
        """Basic dataset validation checks."""
        if df.empty:
            raise ValueError("Dataset is empty.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns.")
        missing_pct = df.isnull().mean().mean() * 100
        if missing_pct > 50:
            print(f"[Loader] WARNING: {missing_pct:.1f}% of values are missing.")
        print(f"[Loader] Validation passed — {missing_pct:.1f}% missing values.")

    @staticmethod
    def _version(path: str):
        """Save a versioned copy of the dataset."""
        os.makedirs(DatasetLoader.VERSIONS_DIR, exist_ok=True)
        with open(path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()[:8]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(path)
        dest = os.path.join(DatasetLoader.VERSIONS_DIR, f"{ts}_{digest}_{filename}")
        shutil.copy2(path, dest)
        print(f"[Loader] Version stored → {dest}")
