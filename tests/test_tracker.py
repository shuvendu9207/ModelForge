"""Tests for ExperimentTracker."""
import unittest
import os, tempfile

from src.tracking.tracker import ExperimentTracker


class TestTracker(unittest.TestCase):

    def setUp(self):
        self.db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db.close()
        self.tracker = ExperimentTracker(db_path=self.db.name)

    def tearDown(self):
        os.unlink(self.db.name)

    def test_log_and_load(self):
        exp_id = self.tracker.log(
            dataset="data.csv",
            model_type="random_forest",
            params={"n_estimators": 100},
            metrics={"accuracy": 0.95, "f1": 0.94},
            training_time=1.23,
        )
        exp = self.tracker.load(exp_id)
        self.assertEqual(exp["model"], "random_forest")
        self.assertAlmostEqual(exp["metrics"]["accuracy"], 0.95)

    def test_load_all(self):
        self.tracker.log("d.csv", "svm", {}, {"accuracy": 0.80})
        self.tracker.log("d.csv", "xgboost", {}, {"accuracy": 0.90})
        all_exp = self.tracker.load_all()
        self.assertEqual(len(all_exp), 2)


if __name__ == "__main__":
    unittest.main()
