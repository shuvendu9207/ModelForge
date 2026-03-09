"""Tests for Cleaner, Encoder, and Scaler."""
import unittest
import pandas as pd

from src.preprocessing.cleaner import Cleaner
from src.preprocessing.encoder import Encoder
from src.preprocessing.scaler  import Scaler


class TestCleaner(unittest.TestCase):

    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        result = Cleaner().fit_transform(df)
        self.assertEqual(len(result), 1)

    def test_fills_missing_mean(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = Cleaner(strategy="mean").fit_transform(df)
        self.assertFalse(result["a"].isnull().any())


class TestEncoder(unittest.TestCase):

    def test_onehot(self):
        df = pd.DataFrame({"cat": ["A", "B", "A"], "val": [1, 2, 3]})
        result = Encoder(method="onehot").fit_transform(df)
        self.assertGreater(len(result.columns), 1)


class TestScaler(unittest.TestCase):

    def test_standard_scale(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = Scaler(method="standard").fit_transform(df)
        self.assertAlmostEqual(result["a"].mean(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
