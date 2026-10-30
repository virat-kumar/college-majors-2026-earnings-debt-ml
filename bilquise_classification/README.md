# Bilquise: classification model (does a program beat its field's national median?)

A companion classification study to the group's earnings regression. Same source data, but the target is yes/no:
does a program's 4-year median earnings exceed the **national median for the same field and credential**
(`earnings_vs_national_pct > 0`, 47% positive)?

## Files
- `analysis_class_only.ipynb`: full analysis in five sections (Motivation, Data, Model, Model Comparison, Conclusion)
- `figures/`: EDA, ROC/PR curves, confusion matrices, permutation importance
- `report/Project_Report_ClassOnly.docx` and `report/Project_Presentation_ClassOnly.pptx`: write-up and slides

## Preprocessing
- **Rows:** keep programs that report earnings, then remove 6,245 duplicate multi-campus rows that repeat one shared
  outcome (key: `opeid6`, `cip_code_4digit`, `credential_level`), leaving 51,867 programs at 4,672 schools.
- **Leakage:** drop every post-graduation and earnings-derived column. Only pre-enrollment information is used.
- **Split:** `GroupShuffleSplit` 80/20 by `opeid6`, so test schools are never seen in training. Tuning uses `GroupKFold(3)`.
- **Features:** log1p of skewed counts, winsorizing at the 1st/99th percentiles (caps learned on training data only),
  missing-value indicators, median imputation, `StandardScaler`, `OneHotEncoder(min_frequency=20)` in one pipeline.

## Models (all covered in lectures) and test results on unseen schools

| Model | Test ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|
| **XGBoost** | **0.781** | 0.703 | 0.673 |
| Random Forest | 0.778 | 0.700 | 0.668 |
| Logistic Regression (L1/LASSO vs L2/Ridge; L2 won) | 0.735 | 0.673 | 0.640 |
| Decision Tree (CART, cost-complexity pruning) | 0.698 | 0.646 | 0.618 |

Majority-class baseline accuracy: 0.527. Top drivers: school location, enrollment, out-of-state tuition, the field's
national 25th-percentile earnings and credential level.

## Run it
From the repository root, run `python setup.py` (downloads `data/college_majors_2026.csv`), then
open `bilquise_classification/analysis_class_only.ipynb` and run all cells (about 15 minutes; needs `xgboost`).

## Note on this branch's split
This study uses its own institution-grouped split (all programs with reported earnings, every credential level), not
the group's `data/split_ids.csv` bachelor's-only regression split, so its numbers are not directly comparable to the
group's MAE results.

## GenAI use
Code, text and slides were drafted with Claude (Anthropic) and are marked per the course GenAI policy.
