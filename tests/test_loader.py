"""Tests for DatasetLoader."""
import unittest
import pandas as pd
import os, tempfile

from src.loader.dataset_loader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        self.tmp.write("feature1,feature2,label\n1,2,0\n3,4,1\n5,6,0\n")
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_load_csv(self):
        df = DatasetLoader.load(self.tmp.name)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            DatasetLoader.load("nonexistent.csv")

    def test_unsupported_format(self):
        with self.assertRaises(ValueError):
            DatasetLoader.load("file.xyz")


if __name__ == "__main__":
    unittest.main()
