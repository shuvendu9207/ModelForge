from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:

    def __init__(self, method="variance", n_features=None):

        self.method = method
        self.n_features = n_features
        self.selector = None

    def fit_transform(self, X, y=None):

        if self.method == "variance":

            self.selector = VarianceThreshold()

            X_new = self.selector.fit_transform(X)

            print(f"[Selector] Selected {X_new.shape[1]} features via '{self.method}'.")

            return X_new

        return X

    def transform(self, X):

        if self.selector:
            return self.selector.transform(X)

        return X