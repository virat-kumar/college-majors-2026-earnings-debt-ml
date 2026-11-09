# Predicting College Program Return on Investment
### BUAN 6341 · Applied Machine Learning · Fall 2026 · Group 2

**What this project actually does:** predicts **median annual earnings four years after completing a bachelor's program at a specific U.S. institution**, using that program's major, college context, geography, national field benchmark, and linked labor-market signals. It helps students and families question generic major-wide earnings headlines and compare *program-specific* financial outlooks. **It does not calculate net ROI, predict an individual's future paycheck, or establish causation.**

This repository documents the research from raw public data through cleaning, exploratory analysis, model tuning, school-disjoint testing, saved prediction pipelines, and two plain-English inference stories. For structured, machine-readable project metadata, see [readme.toml](readme.toml).

## At a glance

| Scope | Verified project statistic |
|---|---:|
| Raw source | **227,980 rows × 72 columns** |
| Clean bachelor's program sample with reported earnings | **25,283 programs** |
| Institutions in final analytical sample | **1,799** |
| Training partition | **19,932 programs / 1,439 institutions** |
| Held-out test partition | **5,351 programs / 360 institutions** |
| Preprocessing input schema | **31 features (25 numeric, 6 categorical)** |
| Encoded model feature count | **88** |
| Target | `target_log_earnings_4yr` |
| Prediction shown to a person | `exp(model.predict(...))` → **USD annual median earnings** |
| Best overall model | **Random Forest** |
| Random Forest test MAE | **$6,232** |
| Random Forest median absolute percentage error | **7.6%** |

The unit of observation is an *institution × academic field × bachelor's credential* program. The test split is **by institution**, not by row: no school is in both the training and held-out test set. The reported outcomes cover federally aided graduates and exclude small programs where earnings data are suppressed.

## Project layout

```text
college-majors-2026-earnings-debt-ml/
├── README.md                       ← Start here; human-readable project guide
├── readme.toml                     ← Machine-readable structure, metrics, paths
├── requirements.txt               ← Python and notebook dependencies
├── .gitignore / .gitattributes
│
├── 0. Submission/
│   └── College_Majors_2026_Model_Comparison_and_Conclusion.docx
│                                  ← Written project submission
│
├── 1. Data/
│   ├── README.md                   ← Data licensing, download, checksums
│   ├── source_manifest.json        ← Raw dataset provenance and SHA-256
│   └── split_ids.csv               ← Versioned program/school split references
│
├── 2. EDA/
│   ├── College_Majors_Bachelors_Data_Prep_EDA.ipynb
│   │                              ← Raw-data verification, filtering,
│   │                                 cleaning, leakage checks, EDA and export
│   ├── bachelors_cleaning_log.csv  ← Auditable row-removal summary
│   ├── bachelors_model_ready.csv   ← 25,283 model-ready program records
│   └── figures/                    ← Exploratory visualizations
│
├── 3. Modelling/
│   ├── College_Majors_Model1_Ridge.ipynb
│   ├── College_Majors_Model2_RandomForest.ipynb
│   ├── College_Majors_Model3_GradientBoosting.ipynb
│   ├── figures/                    ← Model comparison, error/importance plots
│   └── saved_models/
│       ├── ridge_pipeline.joblib
│       ├── random_forest_pipeline.joblib
│       └── gradient_boosting_pipeline.joblib
│                                  ← Fitted preprocessing + estimator pipelines
│
├── 4. Inference/
│   └── College_Program_Inference_Two_Stories.ipynb
│                                  ← Exactly two narrative demo scenarios;
│                                     loads saved models, no retraining
│
└── tests/
    └── test_inference_smoke.py     ← Automated schema/split/serialization
                                       and held-out scenario smoke checks
```

**Raw source CSV:** `college_majors_2026.csv` (~126 MB) is deliberately **not stored as an ordinary Git blob**. The exact upstream snapshot is published in [data-v1.0.0 GitHub Release](https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml/releases/tag/data-v1.0.0); the source SHA-256 is recorded in `1. Data/source_manifest.json`. The **model-ready CSV, canonical split IDs, and trained pipelines** are intentionally versioned with the project so the inference demo works without retraining.

## Workflow: raw data → a decision a person can understand

1. **Source and verify:** Download `college_majors_2026.csv`, verify its SHA-256, and preserve institution/program IDs as strings. The EDA notebook can locate or fetch the source.
2. **Clean and prevent target leakage:** Keep bachelor's programs with valid four-year earnings, handle suppressions, duplicates and anomalous values, and **exclude target-derived and future-outcome columns**. Record cleaning counts.
3. **Build 31 pre-outcome predictors:** Program/college size, location, tuition, admission/SAT context, national field median, linked occupation employment/wage/growth, and occupational AI indicators. The model learns from 25 numeric and six categorical input columns.
4. **Create one school-disjoint split:** 19,932 train and 5,351 held-out test programs; use **GroupKFold by school** for 5-fold model selection. This better reflects prediction for a school unseen during training.
5. **Train and compare:** Field-median baseline, Ridge Regression, Random Forest, and Histogram Gradient Boosting, scored on untouched test schools.
6. **Save complete pipelines:** `*.joblib` contains the fitted preprocessing and the fitted model. Inference expects the **31 original feature columns**; preprocessing and category encoding happen internally.
7. **Infer and explain:** A person supplies **school + bachelor's major + their question/goal**. A program record supplies the remaining 31 stored characteristics. The application feeds **only those features** into the model, transforms the log-scale prediction to dollars, and explains its meaning and limitations. Recorded earnings are **never an input**; they are revealed only for a retrospective accuracy check in the demo.

### Model comparison — held-out schools

| Model | CV R² (log) | Test R² (log) | Test MAE | Test RMSE | Median abs. % error |
|---|---:|---:|---:|---:|---:|
| National field median baseline | — | 0.566 | $8,760 | $12,624 | 10.4% |
| Ridge Regression | 0.753 | 0.744 | $7,030 | $10,005 | 8.6% |
| **Random Forest (best)** | **0.789** | **0.786** | **$6,232** | **$8,914** | **7.6%** |
| Histogram Gradient Boosting | 0.788 | 0.783 | $6,314 | $8,957 | 7.7% |

These statistics reflect the experiment's documented held-out evaluation, not guaranteed per-program performance. The model can under-predict very high earners and is not an individualized career or financial forecast.

## The two inference stories (no extra diagrams)

Run **`4. Inference/College_Program_Inference_Two_Stories.ipynb`**. Its two scenario code cells print the full stakeholder conversation and **all 31 exact input fields** actually passed into the stored model:

| Decision story | National field benchmark | Random Forest prediction | Historical held-out median | Model error in *selected* case |
|---|---:|---:|---:|---:|
| Student considering **Computer Science at Western Michigan University** | $107,009 | **$91,424** | $90,876 | 0.6% |
| Family shortlisting **Nursing at California State University–Fresno** | $88,910 | **$113,046** | $111,278 | 1.6% |

The first illustrates why a national major-wide figure can be **too high for a specific school**; the second illustrates why it can **understate a particular program's outlook**. These examples were deliberately selected for close held-out predictions and **do not characterize typical accuracy**; refer to the full evaluation table above.

### Environment and reproducible execution

Tested on Python **3.12**, in the project's JupyterLab Conda kernel. Use the same environment for saved model files: **joblib/pickle objects should only be loaded from trusted sources, and scikit-learn version changes can break compatibility**.

```bash
git clone https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml.git
cd college-majors-2026-earnings-debt-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name college-majors-ml --display-name "College Majors ML"
jupyter lab
```

**Quick demo:** open `4. Inference/College_Program_Inference_Two_Stories.ipynb`, select the newly registered kernel, and **Run All**. The inference notebook loads `3. Modelling/saved_models/*.joblib` and `2. EDA/bachelors_model_ready.csv` from the repository and **does not retrain anything**.

**Rebuild from raw data:** place the verified `college_majors_2026.csv` in the project root or `2. EDA/` (the EDA notebook can also fetch the public dataset), open the EDA notebook from `2. EDA/` and Run All. Then open each modeling notebook in `3. Modelling/` and Run All (training/cross-validation may be **slow**). Finally rerun the inference notebook. Model notebooks assume a working directory of `3. Modelling/` and load the model-ready CSV from `../2. EDA/bachelors_model_ready.csv`.

The notebooks write outputs into their respective directories. If using VS Code Remote SSH, **select the Python kernel associated with the installed dependencies**; do not confuse the VS Code Jupyter widget's CDN warning with a model-training error.

### Verification and test locations

**Automated:** run from the repository root:
```bash
python -m unittest discover -s tests -v
```
The tests verify (1) the expected data/split schema, (2) no school overlap between train/test partitions, (3) that all three saved pipelines deserialize and make finite predictions, (4) that labels/outcomes are not passed as features, and (5) the two deliberately selected held-out inference examples. **No automated full retraining/CI suite is claimed.**

**Notebook checks:** each modeling notebook ends with a save → reload → matching-predictions assertion. The `4. Inference` notebook has a clean-kernel run with one setup and two scenario code cells. Visual EDA/model results are under `2. EDA/figures/` and `3. Modelling/figures/`; the new inference notebook intentionally does **not** create diagrams.

**Reference files:** `1. Data/source_manifest.json` (source integrity), `1. Data/split_ids.csv` (canonical split snapshot), `2. EDA/bachelors_cleaning_log.csv` (cleaning statistics).

## Attribution, limitations, and responsible interpretation

Dataset: [College Majors 2026: Earnings, Debt, Jobs, AI](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai), published by **kylefengkfeng209**. The Kaggle listing identifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) for the data; preserve source attribution and the applicable share-alike terms for derived datasets.

The model predicts a **program-wide historical four-year median** for graduates who received federal aid. It cannot predict an individual's salary, determine whether a particular person will be admitted or employed, or identify a causal return from changing schools. **Tuition alone is not net ROI**: total expenses, scholarships, debt, taxes, employment probability and many personal circumstances are out of scope. Privacy-suppressed small cohorts are excluded; atypically high-earning programs are harder to forecast. Use the system to guide further research, not as a guarantee.
