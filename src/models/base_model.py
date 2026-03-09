"""
Base Model Interface
All ModelForge models inherit from this class.
"""

from abc import ABC, abstractmethod
import pickle
import os


class BaseModel(ABC):

    def __init__(self, params: dict = None):
        self.params = params or {}
        self._model = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def predict_proba(self, X):
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        print(f"[Model] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        print(f"[Model] Loaded from {path}")
