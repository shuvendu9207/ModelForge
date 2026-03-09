import pandas as pd


class CategoricalEncoder:

    def __init__(self):
        self.columns = None

    def fit_transform(self, df):

        df = pd.get_dummies(df, drop_first=True)

        self.columns = df.columns

        return df

    def transform(self, df):

        df = pd.get_dummies(df, drop_first=True)

        df = df.reindex(columns=self.columns, fill_value=0)

        return df