"""Integration test for the full pipeline."""
import unittest
import pandas as pd
import os, tempfile


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w")
        # Generate synthetic classification data
        import random
        self.tmp.write("f1,f2,f3,label\n")
        for _ in range(100):
            vals = [random.uniform(0, 1) for _ in range(3)]
            label = int(vals[0] > 0.5)
            self.tmp.write(",".join(map(str, vals)) + f",{label}\n")
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_pipeline_returns_metrics(self):
        from src.pipeline.pipeline import Pipeline
        p = Pipeline(dataset_path=self.tmp.name, model_type="random_forest")
        metrics = p.execute()
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
