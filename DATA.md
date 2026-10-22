# Dataset acquisition and reproducibility

This repository **does not contain the raw dataset**. Every collaborator downloads the same public Kaggle source locally.

## One-command route (preferred)

From the repository root, with Python 3.10–3.13 installed:

```sh
python setup.py
```

This creates `.venv`, installs `requirements.txt`, downloads the Kaggle dataset with `kagglehub`, verifies the CSV, and writes it to `./college_majors_2026.csv`. No API credential should be committed or shared. Public dataset downloads usually work without Kaggle sign-in; if a policy, account or rate-limit gate prevents it, use the manual route.

## Manual route (no Kaggle API necessary)

1. Open [the College Majors 2026 dataset on Kaggle](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai).
2. Download the dataset archive through Kaggle. You might be asked to accept site terms or sign in.
3. Extract **`college_majors_2026.csv`** from the downloaded archive.
4. Place that file **next to** `College_Majors_EDA_Colab.ipynb`, at the project root (not inside `reports/`, `.venv/` or `data/`).
5. Run `python setup.py` again. The script recognizes the CSV, verifies it, skips a redundant download, and records provenance in `data/source_manifest.json` (ignored).

```text
project-root/
├── College_Majors_EDA_Colab.ipynb
└── college_majors_2026.csv
```

The original Colab notebook finds a CSV at this path automatically, with **zero notebook modifications**.

## Data validation

The download step checks:

- Exact required filename: `college_majors_2026.csv`.
- Schema: **72 columns**, with `program_id` as the initial column for the referenced version.
- Expected dataset size: **227,980 data rows** (header excluded).
- Local **SHA256** digest, written to untracked `data/source_manifest.json`.

If Kaggle publishes an updated version with a new schema/row count, do not silently overwrite the project baseline. Review the source dataset and update the expected schema in a reviewed PR after validating downstream analysis.

## Scientific considerations

- Rows are institution/field/credential program records, not individual student observations.
- Outcome availability is privacy-suppressed. Suppression is not random; do not impute the response and report results as representative of every program.
- The notebook saves a group-wise **institution split** in `split_ids.csv`, also ignored from Git; use it when comparing different models on the same split.
- Only preprocess and fit imputation/scaling/encoding on training folds, not on full data, to avoid leakage.
- Kaggle metadata describes this dataset as **CC BY-SA 4.0**. Include attribution to [the original listing](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai) when using it in class reports. Do not assume this grants permission to expose private data or unrelated credentials.

## Troubleshooting

**`python` not recognized:** Install [Python](https://www.python.org/downloads/) and enable it on PATH. On Linux use `python3 setup.py`; Windows can use `py -3.13 setup.py`. Avoid Python 3.14 until dependencies are verified.

**Ubuntu says `ensurepip` missing:** Install the matching `python3-venv` package through your package manager, then repeat setup.

**Kaggle download errors:** Check network access and visit the dataset page manually. The dataset is public, but Kaggle can still enforce rate limits, authentication or terms acceptance. Do not put Kaggle tokens in this repository.

**Unexpected CSV schema:** Ensure you downloaded the exact dataset, not a different College Scorecard file. Do not disable checks merely to force an incompatible file through.

**Notebook cannot locate the CSV:** Start Jupyter with `python launch.py`, which sets the working directory to the project root. Put the CSV next to the notebook, not in `Downloads`.

**The original notebook created new CSVs:** This is expected. `.gitignore` keeps `*.csv` untracked; share dataset IDs or derived results through an approved course-approved data channel, not GitHub.
