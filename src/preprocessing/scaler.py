"""
Feature Scaler
Scales numeric features using Standard or MinMax scaling.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaler:

    def __init__(self, method: str = "standard"):
        """method: 'standard' | 'minmax' | 'none'"""
        self.method = method
        self._scaler = None

    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        numeric_cols = [c for c in df.select_dtypes(include="number").columns
                        if c != target_col]
        if not numeric_cols or self.method == "none":
            return df

        if self.method == "standard":
            self._scaler = StandardScaler()
        elif self.method == "minmax":
            self._scaler = MinMaxScaler()

        df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols])
        print(f"[Scaler] Applied '{self.method}' scaling to {len(numeric_cols)} columns.")
        return df
