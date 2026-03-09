from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ModelTrainer:

    def __init__(self, model_type="random_forest", params=None):

        self.model_type = model_type
        self.params = params or {}

        self.model = self._build_model()

    def _build_model(self):

        if self.model_type == "random_forest":
            return RandomForestClassifier(**self.params)

        elif self.model_type == "logistic_regression":
            return LogisticRegression(**self.params)

        elif self.model_type == "svm":
            return SVC(**self.params)

        else:
            raise ValueError(f"Unknown model: {self.model_type}")

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)