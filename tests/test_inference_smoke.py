"""Smoke tests for versioned college-major dataset and trusted fitted pipelines.

Run from repo root: python -m unittest discover -s tests -v
These tests deliberately do NOT retrain any model or touch remote services.
"""
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "2. EDA" / "bachelors_model_ready.csv"
MODEL_DIR = ROOT / "3. Modelling" / "saved_models"
MODELS = {
    "ridge": "ridge_pipeline.joblib",
    "random_forest": "random_forest_pipeline.joblib",
    "gradient_boosting": "gradient_boosting_pipeline.joblib",
}
SCENARIOS = {
    "172699-11.07-3": 90876,  # Western Michigan University: Computer Science
    "110556-51.38-3": 111278,  # California State University-Fresno: Nursing
}
NO_LEAK = {
    "median_earnings_4yr_usd", "target_log_earnings_4yr",
    "target_above_national", "program_id", "log_median_debt"
}


class InferenceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = pd.read_csv(
            DATA_PATH, dtype={"opeid6": str, "cip_code_4digit": str}
        )
        cls.models = {key: joblib.load(MODEL_DIR / path) for key, path in MODELS.items()}
        cls.train = cls.dataset.loc[cls.dataset.split.eq("train")]
        cls.test = cls.dataset.loc[cls.dataset.split.eq("test")]

    def test_data_volume_and_group_disjointness(self):
        self.assertEqual(len(self.dataset), 25283)
        self.assertEqual(len(self.train), 19932)
        self.assertEqual(len(self.test), 5351)
        self.assertEqual(self.dataset.opeid6.nunique(), 1799)
        self.assertEqual(self.train.opeid6.nunique(), 1439)
        self.assertEqual(self.test.opeid6.nunique(), 360)
        self.assertTrue(set(self.train.opeid6).isdisjoint(self.test.opeid6))

    def test_models_deserialize_and_predict_without_leakage(self):
        columns = None
        for name, model in self.models.items():
            with self.subTest(model=name):
                self.assertIn("prep", model.named_steps)
                self.assertIn("model", model.named_steps)
                features = list(model.named_steps["prep"].feature_names_in_)
                self.assertEqual(len(features), 31)
                self.assertFalse(set(features) & NO_LEAK)
                self.assertTrue(set(features).issubset(self.dataset.columns))
                if columns is not None:
                    self.assertEqual(columns, features)
                else:
                    columns = features
                estimates = np.exp(model.predict(self.test[features].iloc[:6]))
                self.assertTrue(np.isfinite(estimates).all())
                self.assertTrue((estimates > 0).all())

    def test_two_genuine_held_out_examples(self):
        model = self.models["random_forest"]
        features = list(model.named_steps["prep"].feature_names_in_)
        for program_id, historical in SCENARIOS.items():
            with self.subTest(program=program_id):
                match = self.test.loc[self.test.program_id.eq(program_id)]
                self.assertEqual(len(match), 1)
                self.assertNotIn(match.iloc[0]["opeid6"], set(self.train.opeid6))
                self.assertEqual(int(match.iloc[0]["median_earnings_4yr_usd"]), historical)
                predicted = float(np.exp(model.predict(match[features])[0]))
                self.assertLess(abs(predicted - historical) / historical, .05)


if __name__ == "__main__":
    unittest.main()
