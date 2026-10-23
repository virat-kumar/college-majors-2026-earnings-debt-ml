"""Fast, Windows/Linux unit checks for published artifacts and source provenance."""
import hashlib
import json
import re
from pathlib import Path
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"


class PublishedArtifactTests(unittest.TestCase):
    def test_four_family_comparison_has_five_folds_each(self):
        summary=pd.read_csv(REPORTS/"model_comparison.csv")
        folds=pd.read_csv(REPORTS/"model_selection_folds.csv")
        self.assertEqual(set(summary.model), {"dummy","ridge","decision_tree","random_forest","hist_gradient_boosting"})
        self.assertEqual(len(folds),25)
        for _, group in folds.groupby("model"):
            self.assertEqual(set(group.fold),set(range(5)))

    def test_holdout_metrics_and_tuning_selection(self):
        d=json.loads((REPORTS/"model_evaluation.json").read_text())
        self.assertEqual(d["selected_family"],"hist_gradient_boosting")
        self.assertEqual(d["train_programs"],19512)
        self.assertEqual(d["test_programs"],5771)
        self.assertEqual(len(pd.read_csv(REPORTS/"tuning_candidates.csv")),12)
        self.assertLess(d["holdout"]["tuned_winner"]["mae_usd"],d["holdout"]["dummy_mean"]["mae_usd"])

    def test_final_pdf_and_all_markdown_figures(self):
        p=ROOT/"submission"/"Group_2_Final_Project_Report.md"
        q=p.with_suffix(".pdf")
        self.assertTrue(q.is_file() and q.stat().st_size>200_000)
        self.assertEqual(q.read_bytes()[:5],b"%PDF-")
        images=re.findall(r"!\[[^\]]*\]\(([^)]+)\)",p.read_text())
        self.assertEqual(len(images),15)
        for image in images:
            with self.subTest(image=image):
                self.assertTrue((p.parent/image).is_file())

    def test_original_eda_extracted_graphics_and_no_workflows(self):
        d=json.loads((REPORTS/"eda_figure_manifest.json").read_text())
        self.assertEqual(len(d["figures"]),12)
        for e in d["figures"]:
            with self.subTest(figure=e["figure"]):
                data=(ROOT/e["figure"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(),e["sha256"])
        self.assertFalse(list((ROOT/".github"/"workflows").glob("*")))


if __name__=="__main__":
    unittest.main()
