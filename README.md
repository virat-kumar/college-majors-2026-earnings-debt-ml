# College Majors 2026: Earnings, Debt Risk & Career Outcomes

**BUAN 6341 · Applied Machine Learning · Group 2 · The University of Texas at Dallas**

A reproducible, institution-aware machine learning study of **college program earnings**, **student debt risk**, and **career outlook / AI exposure** using the *College Majors 2026: Earnings, Debt, Jobs, AI* dataset.

> **Project status:** Data cleaning, exploratory data analysis, leakage audit, grouped data split, preprocessing and a regression baseline are available. Final model selection and classification scope remain group decisions. Reported EDA associations are **not causal claims**.

## Start in three steps

Install **Python 3.10–3.13** and [Git](https://git-scm.com/), then in a terminal (PowerShell on Windows or bash on Linux):

```sh
git clone https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml.git
cd college-majors-2026-earnings-debt-ml
python setup.py
```

**The same `python setup.py` command works on Windows and Linux.** It:

1. Creates a project-local, isolated `.venv` using your Python interpreter (no system package installation).
2. Installs packages from `requirements.txt`.
3. Downloads the original dataset from its public Kaggle page, checks the expected schema and row count, and copies `college_majors_2026.csv` into the repository root **locally only**.
4. Smoke-tests installed dependencies and confirms the notebook can find the dataset.

**Launch** after setup:

```sh
python launch.py
```

JupyterLab opens in your browser. Open **`College_Majors_EDA_Colab.ipynb`**, then choose **Run All** from the top. You do **not** have to activate a virtual environment manually.

If your Linux distro only provides the `python3` command, run `python3 setup.py` and `python3 launch.py` instead. If Windows uses the Python Launcher, `py -3.13 setup.py` is equivalent. Setup refuses unsupported interpreter versions rather than modifying your system. On Ubuntu/Debian, install the corresponding `python3-venv` package if venv creation is unavailable.

## Project structure

```text
college-majors-2026-earnings-debt-ml/
├── College_Majors_EDA_Colab.ipynb  # Original, unedited EDA notebook
├── README.md                       # This guide
├── DATA.md                         # Dataset provenance, manual installation, version checks
├── CONTRIBUTING.md                 # Team Git + review guidelines
├── requirements.txt                # Python dependencies
├── setup.py                        # Cross-platform, one-command full setup
├── download_data.py                # Download + validate dataset, never commit
├── validate.py                     # Smoke tests; optional notebook run
├── launch.py                       # Cross-platform JupyterLab launcher
├── reports/
│   └── EDA_SUMMARY.md              # Findings from group's 2-page summary
├── .github/workflows/ci.yml        # Linux / Windows setup validation
├── .gitignore                      # Explicit dataset/output/secret safeguards
│
├── .venv/                          # Created by setup, IGNORED
├── college_majors_2026.csv         # Downloaded original CSV, IGNORED
├── split_ids.csv                   # Created by original notebook, IGNORED
└── data/source_manifest.json      # Local SHA256 + provenance, IGNORED
```

**The raw CSV, downloaded files, row-level splits, Jupyter execution artifacts, caches, models, private tokens and environments are never committed.** Clone the code, then download the data with the setup script.

## Research questions

| Task | Target | Status / evaluation |
| --- | --- | --- |
| **Regression** | `median_earnings_4yr_usd`: 4-year median earnings after completion | Baseline: mean-only grouped-CV **MAE $15,799** for bachelor's programs; compare better models using MAE (and inspect dollar-scale error if modeling log earnings) |
| **Classification** | High debt-risk using `payment_to_income_pct_1yr > 8` | Threshold and whether to retain this problem across **all credential levels** require group sign-off; avoid target leakage |
| **Career analysis** | Job outlook and AI-software exposure | Exploratory association only, not a causal effect of AI on earnings |

The notebook contains the initial **data cleaning, EDA and shared split**, not a complete final supervised modeling pipeline. Contributors should add model experiments as new notebooks or modules rather than rewriting the original EDA source.

### Baseline data and reproducibility

The public Kaggle dataset contains **227,980 program records and 72 variables**. Each record represents an institution, study field and credential combination. Original EDA cleaning removes foreign-institution entries and redundant branch-campus copies, yielding **212,707 records**. Target availability is limited by privacy suppression: 4-year earnings are reported for **51,818** programs, and debt-to-income for **37,789** programs. This missingness is not random; do not generalize to all programs without discussion.

For unbiased evaluation, the original notebook makes an **80/20 train/test split by institution (`opeid6`)**. It also defines **five institution-grouped cross-validation folds**, preventing one institution appearing in both training and test sets or both sides of a CV fold. Exploratory comparisons and preprocessing fit on training data only.

**Leakage safeguards:** Do not add other outcome-derived columns as regression predictors. The EDA explicitly flags 1-/5-year earnings, debt-to-earnings and status/target-derived fields, including `national_median_earnings_4yr_usd` and `earnings_vs_national_pct`. Revisit feature selection during group review.

For full EDA findings, see [reports/EDA_SUMMARY.md](reports/EDA_SUMMARY.md).

## Dataset: download, location and license

**Original source:** [College Majors 2026: Earnings, Debt, Jobs, AI — Kaggle](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai)

**Data license:** The Kaggle listing identifies **CC BY-SA 4.0**. Please follow the source attribution/license requirements. This repository deliberately **does not redistribute the dataset**.

```text
# Desired local layout:
college-majors-2026-earnings-debt-ml/
├── College_Majors_EDA_Colab.ipynb
└── college_majors_2026.csv       # Local, ignored
```

`python setup.py` does this automatically. If the Kaggle API is unavailable, download the dataset ZIP through the linked Kaggle page, extract **`college_majors_2026.csv`** and place the CSV in the project root, **next to the notebook**. Re-run `python setup.py` to validate it. See [DATA.md](DATA.md) for troubleshooting, SHA256 manifest, and data version controls.

## Common commands

| Goal | Both Windows and Linux |
| --- | --- |
| Initial or repeat installation + dataset download | `python setup.py` |
| Install only / CI without dataset | `python setup.py --skip-data` |
| Revalidate after installation | `python .venv/Scripts/python.exe validate.py` (Windows) or `./.venv/bin/python validate.py` (Linux) |
| Start JupyterLab | `python launch.py` |
| Re-fetch/check dataset (after setup) | `python setup.py --skip-install` |
| Execute the notebook in a copy (after setup) | `python .venv/Scripts/python.exe validate.py --execute` (Windows) or `./.venv/bin/python validate.py --execute` (Linux) |

The optional `--execute` command does **not** overwrite the checked-in notebook; it writes a run-result copy into ignored `.artifacts/` instead. This notebook creates `split_ids.csv` at runtime; that file also remains private and is ignored.

## Collaboration

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes.

- Create a feature branch named `feature/<your-task>` and send a pull request to `main`.
- Reuse the same **unmodified, audited dataset and institution-level split** for a fair apples-to-apples model comparison.
- Record training population, chosen features, preprocessing, validation folds, baseline and random seed in each PR.
- Do not upload raw data, generated output files, Kaggle tokens, Colab credentials, or environment folders.
- Coordinate decisions on **bachelor's vs. all credentials**, **the debt-risk task/threshold**, **final feature list** and **raw vs. log earnings** before merging modeling results.

**Academic integrity:** Attribute work and data sources; use this shared repository for collaboration and reproducible analysis, and follow applicable course rules.

## Credits and scope

Original data-cleaning/EDA notebook prepared by **Manoj Elangovan Jayanthi**, as credited in the source notebook. Project organized for **BUAN 6341 Group 2** at UT Dallas. Dataset assembled by the Kaggle publisher from College Scorecard, NCES CIP-to-SOC mapping, BLS projections and O*NET. See original dataset page for credits and licensing.

**License note:** Dataset licensing is not the same as code/notebook licensing. No additional license is asserted for teammates' original notebook content without their permission.
