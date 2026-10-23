"""Run with: python -m unittest discover -s tests -v."""
import hashlib
import unittest
import numpy as np
from modeling.common import (
    FEATURES, ROOT, TARGET, EXPECTED_NOTEBOOK_SHA256,
    folds_for, load_study, pipeline_for,
)


class StudyIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = load_study()

    def test_original_notebook_is_byte_identical(self):
        got = hashlib.sha256((ROOT / "College_Majors_EDA_Colab.ipynb").read_bytes()).hexdigest()
        self.assertEqual(got, EXPECTED_NOTEBOOK_SHA256)

    def test_original_group_partition_and_population(self):
        self.assertEqual(len(self.train), 19512)
        self.assertEqual(len(self.test), 5771)
        self.assertTrue(set(self.train.opeid6).isdisjoint(self.test.opeid6))
        self.assertTrue(self.train[TARGET].gt(0).all())

    def test_five_folds_isolate_institutions(self):
        folds = folds_for(self.train)
        self.assertEqual(len(folds), 5)
        for fit_idx, val_idx in folds:
            self.assertTrue(
                set(self.train.iloc[fit_idx].opeid6).isdisjoint(
                    self.train.iloc[val_idx].opeid6
                )
            )
            self.assertTrue(set(self.train.iloc[val_idx].cv_fold).issubset(set(range(5))))

    def test_preprocessing_fits_without_nan_or_inf(self):
        sample = self.train.sample(n=500, random_state=42)
        for name in ("ridge", "decision_tree", "random_forest", "hist_gradient_boosting"):
            fitted = pipeline_for(name).named_steps["preprocess"].fit_transform(sample[FEATURES])
            self.assertEqual(len(fitted), 500)
            self.assertTrue(np.isfinite(fitted).all(), name)

    def test_holdout_does_not_enter_training_folds(self):
        training_ids = set(self.train.program_id)
        holdout_ids = set(self.test.program_id)
        self.assertTrue(training_ids.isdisjoint(holdout_ids))


if __name__ == "__main__":
    unittest.main()
