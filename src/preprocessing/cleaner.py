"""
Data Cleaner
Handles missing values and duplicate rows.
"""

import pandas as pd


class Cleaner:

    def __init__(self, strategy: str = "mean"):
        """
        strategy: 'mean' | 'median' | 'drop'
        """
        self.strategy = strategy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"[Cleaner] Removed {initial_rows - len(df)} duplicate rows.")

        numeric_cols = df.select_dtypes(include="number").columns
        cat_cols     = df.select_dtypes(include="object").columns

        if self.strategy == "drop":
            df = df.dropna()
        elif self.strategy in ("mean", "median"):
            fill_fn = df[numeric_cols].mean() if self.strategy == "mean"                       else df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(fill_fn)
            df[cat_cols]     = df[cat_cols].fillna("UNKNOWN")

        print(f"[Cleaner] Missing values handled via '{self.strategy}' strategy.")
        return df
