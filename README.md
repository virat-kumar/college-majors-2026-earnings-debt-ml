# College Majors 2026 | Applied Machine Learning Group 2

**University of Texas at Dallas - BUAN 6341 - Group 2**

We completed a reproducible applied machine-learning study of **program-level median earnings four years after bachelor's degree completion**. Our project proceeds from the original team's EDA through an independent dataset audit, four different regression model families, cross-validation selection, hyperparameter tuning, explainability and an actionable but cautious business case.

**Status: COMPLETE. The final team-ready PDF, editable Word report, source, model selection, diagrams, evidence and code are published.** The original EDA notebook remains byte-for-byte unchanged. The source dataset is public in GitHub Releases. No GitHub Actions workflows are configured.

## Final submission: open this first

| File | What it contains |
|:---|:---|
| **[Final Group 2 project report (PDF)](submission/Group_2_Final_Project_Report.pdf)** | **26-page team-ready submission** with cover, table of contents, 16 real diagrams/charts, all four models, validation, tuning, test metrics, explainability, business case and references |
| **[Editable final report (Word DOCX)](submission/Group_2_Final_Project_Report.docx)** | **21-page editable companion** with formatted tables, numbered figures, full content, separate cover/contents and group handoff |
| [Final report Markdown source](submission/Group_2_Final_Project_Report.md) | Editable source corresponding to the PDF; regenerates it via Pandoc |
| [Detailed model explanation](reports/MODEL_EXPLANATION.md) | Methodology, before/after metrics, uncertainty, interpretability and error slices |
| [Business case](reports/BUSINESS_CASE.md) | Stakeholder design, pilot phases, KPIs, safety and economic limitations |

## Find every project phase

| Stage | Research question / accomplished work | Read and verify here |
|:---|:---|:---|
| **1 - EDA** | What does the data contain? Why clean it? What do earnings, debt and majors look like? | [Unmodified EDA notebook](College_Majors_EDA_Colab.ipynb); [EDA narrative](reports/EDA_SUMMARY.md); [12 extracted original figures and provenance](reports/eda_figure_manifest.json) |
| **2 - Data audit** | Are cleaning numbers and shared-campus deduplication correct? | [Independent audit script](modeling/eda_audit.py); [summary JSON](reports/eda_audit.json); [cleaning CSV](reports/eda_cleaning_audit.csv); [target coverage CSV](reports/eda_scope_audit.csv) |
| **3 - Data engineering** | Are the original test institutions isolated and predictors leakage-safe? | [Shared modeling pipeline](modeling/common.py); [canonical school split CSV](data/split_ids.csv); [data provenance](DATA.md); [tests](tests/) |
| **4 - Model selection** | Which of four regression families predicts reported bachelor's earnings most accurately? | [Model selection code](modeling/model_selection.py); [comparison CSV](reports/model_comparison.csv); [all 25 fold-level scores](reports/model_selection_folds.csv); [comparison plot](reports/figures/model_comparison.png) |
| **5 - Model tuning** | Which settings improve the selected family without viewing final test outcomes? | [Hyperparameter tuning code](modeling/model_fine_tuning.py); [12 candidates CSV](reports/tuning_candidates.csv); [frozen holdout evaluation JSON](reports/model_evaluation.json); [heldout predictions CSV](reports/heldout_predictions.csv) |
| **6 - Model explainability** | What inputs matter and where does the model err? | [Post-fit diagnostic code](modeling/model_explainability.py); [detailed explanation](reports/MODEL_EXPLANATION.md); [feature importance CSV](reports/feature_importance.csv); [error by segment CSV](reports/error_by_segment.csv); [diagnostic figures](reports/figures/) |
| **7 - Business case** | How could this be used responsibly in student advising or academic program research? | [Business case report](reports/BUSINESS_CASE.md); [full report](submission/Group_2_Final_Project_Report.pdf) |
| **8 - Team submission: PDF + Word** | How do all phases connect into one professional deliverable? | [Final PDF](submission/Group_2_Final_Project_Report.pdf), [editable DOCX](submission/Group_2_Final_Project_Report.docx), [Markdown](submission/Group_2_Final_Project_Report.md), [workflow diagram](reports/figures/study_workflow.png) |

## Model comparison: actual experimental results

All four models used the same **14 EDA-selected predictors**, the same **five institution-disjoint folds**, and USD-scale mean absolute error (MAE) as the primary choice criterion.

| Model | Five-fold validation MAE |
|:---|---:|
| **Histogram Gradient Boosting** (selected family) | **USD 6,880.79** |
| Random Forest | USD 6,918.22 |
| Decision Tree | USD 8,247.56 |
| Ridge Regression | USD 8,964.19 |
| Mean-only baseline (reference, not one of the four) | USD 15,806.18 |

We tuned the selected boosted-tree family using **12 configurations and five training-only folds**. Best tuned CV MAE was **USD 6,855.28**.

Our final test was **5,771 eligible bachelor's-program records from 379 schools never present in training**. The tuned model achieved **USD 6,956.68 test MAE**, **USD 10,111.64 RMSE** and **R² = 0.7793**, while untuned boosting achieved USD 6,993.01 MAE and the mean-only comparator USD 16,137.72 MAE.

**Important:** Tuning improved test MAE only **USD 36.34** over the untuned winner, and a school-clustered bootstrap interval includes zero. We do **not** claim the small tuning advantage is established. The much larger improvement versus a constant-mean predictor reflects useful prediction, **not** student earnings gained or proven causal benefits.

## Data available in this repository

The public [College Majors 2026: Earnings, Debt, Jobs, AI](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai) dataset (Kaggle listing: **CC BY-SA 4.0**) has **227,980 rows x 72 variables**. The EDA cleaned it to **212,707 U.S. institution-program records**. Four-year earnings are reported for **51,818** of those; **25,283** are bachelor's records.

Both exact source CSV snapshots are public:

- **[Original college_majors_2026.csv (126.3 MB) - GitHub Release](https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml/releases/download/data-v1.0.0/college_majors_2026.csv)**. It is intentionally a versioned GitHub Release asset because GitHub regular Git rejects blobs over 100 MiB.
- **[Canonical split_ids.csv (8.9 MB) - tracked directly](data/split_ids.csv)** and [also in that Release](https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml/releases/download/data-v1.0.0/split_ids.csv).
- All **reviewed derived metrics/analysis CSVs** are Git-tracked in [reports/](reports/), including model selection, tuning candidates, program-level heldout predictions, and explainability results.

[DATA.md](DATA.md) records provenance, exact SHA-256 checksums and licensing. We do not publish passwords, Kaggle keys, virtual environments, caches or arbitrary local CSV files.

## Set up and reproduce (Windows or Linux)

Install Python **3.10-3.13** and Git. Then:

```bash
git clone https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml.git
cd college-majors-2026-earnings-debt-ml
python setup.py
```

The same `python setup.py` creates `.venv`, installs `requirements.txt`, automatically downloads and SHA-256-verifies the source CSV from our GitHub Release (or falls back to Kaggle), and validates the notebook imports. On systems that call Python `python3`, use `python3 setup.py`.

To repeat **every stage** using the correct isolated interpreter without manually activating it:

```bash
python run_project.py
```

This runs the EDA integrity audit, exact original EDA figure extraction, four-model comparison, grouped hyperparameter search, final holdout evaluation, explainability and correctness tests. It takes longer than setup because models are genuinely retrained. To check the already-computed committed outputs without retraining:

```bash
python run_project.py --verify-only
```

To explore the original notebook in JupyterLab, use `python launch.py`. To regenerate the same PDF from Markdown (requires Pandoc and XeLaTeX, **not** needed to read or submit the committed PDF), run `python submission/render_report.py`.

## Project organization

```text
college-majors-2026-earnings-debt-ml/
├── College_Majors_EDA_Colab.ipynb  # Original, unchanged EDA
├── README.md, DATA.md, CONTRIBUTING.md
├── setup.py, requirements.txt, run_project.py
├── download_data.py, launch.py, validate.py
├── data/
│   ├── split_ids.csv                # The original, canonical group split
│   └── README.md
├── modeling/
│   ├── common.py                    # 14 predictors, canonical folds, pipelines
│   ├── eda_audit.py                 # Independent EDA audit
│   ├── extract_eda_figures.py       # 12 original notebook images
│   ├── model_selection.py           # Four regression families
│   ├── model_fine_tuning.py         # 12-setting grouped search, final test
│   ├── model_explainability.py      # Permutation importance and errors
│   └── README.md
├── reports/
│   ├── EDA_SUMMARY.md, MODEL_EXPLANATION.md, BUSINESS_CASE.md
│   ├── *.csv, *.json                # Executed, auditable evidence
│   └── figures/                    # Original EDA + new model charts
├── submission/
│   ├── Group_2_Final_Project_Report.md
│   ├── Group_2_Final_Project_Report.pdf
│   ├── Group_2_Final_Project_Report.docx
│   ├── README.md, render_report.py, render_word.py, make_workflow_diagram.py, report_style.tex
├── tests/
└── .venv/, .artifacts/, downloaded raw CSV  # local-only, ignored
```

**No GitHub Actions workflows:** We intentionally removed the original `.github/workflows/ci.yml` and did not restore it.

## Research limitations and attribution

Our final prediction target is reported **bachelor's program-level** median earnings, **not individual graduate pay**. Outcome suppression is extensive; no result proves effects of changing a major, tuition, curriculum or AI exposure. Debt-to-income classification was *explored in EDA* but **not trained as a classifier** here. The full PDF discusses institutional/group biases, data timing, error distribution and proposed supervised pilot safeguards.

The protected team EDA notebook credits **Manoj Elangovan Jayanthi** as its preparer. We retained its source bytes and original graphics, and credit Kaggle dataset publisher **kylefengkfeng209**. The source dataset and reviewed derived split are attributed under the published dataset's CC BY-SA 4.0 terms; these do not automatically license teammates' original software/notebook content.
