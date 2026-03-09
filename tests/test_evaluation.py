"""Tests for Evaluator."""
import unittest
import numpy as np

from src.evaluation.metrics import Evaluator


class TestEvaluator(unittest.TestCase):

    def test_perfect_predictions(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        metrics = Evaluator().compute(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_wrong_predictions(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        metrics = Evaluator().compute(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
