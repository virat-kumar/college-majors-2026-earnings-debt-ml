# Data provenance and reproducible snapshots

The project uses the public **College Majors 2026: Earnings, Debt, Jobs, AI** dataset by **kylefengkfeng209**, downloaded from [Kaggle](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai).

**Original raw source:** `college_majors_2026.csv` — **227,980 rows × 72 columns** and **126,340,696 bytes**.
- [Versioned GitHub Release: data-v1.0.0](https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml/releases/tag/data-v1.0.0)
- [Direct download of raw CSV](https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml/releases/download/data-v1.0.0/college_majors_2026.csv)
- Original file SHA-256: `58f1c10e79793f33049f3b381cd81d4d0c190ca1052e427c29779d0c89c5ceca`
- Machine-readable provenance: [source_manifest.json](source_manifest.json)

The 126 MB raw file is **not checked into ordinary Git**. The EDA notebook verifies the source's SHA-256 and can obtain the public Kaggle file when missing. Place the original file at the repository root or inside `2. EDA/` if you download it manually.

**The Git-tracked split reference:** [split_ids.csv](split_ids.csv) — **212,707 program-level rows**, 7 columns: `program_id`, `opeid6`, `credential_level`, `split`, `cv_fold`, `in_regression_set`, `in_classification_set`.
- SHA-256: `e5215d887840ae25e0c953793842b1886670f51b879783904988a1348f7bfe6c`
- This is the original versioned program split reference; the modeling notebooks use the finalized cleaned train/test assignments recorded in `2. EDA/bachelors_model_ready.csv`.

**Cleaned model-ready table:** [bachelors_model_ready.csv](../2.%20EDA/bachelors_model_ready.csv) — **25,283 bachelor's program records**, representing **1,799 institutions**. Contains the 31 predictors, partition labels, and outcome columns kept separate during model prediction. Cleaning counts are in [bachelors_cleaning_log.csv](../2.%20EDA/bachelors_cleaning_log.csv).

## Attribution and license

The source dataset is labeled **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** on Kaggle: https://creativecommons.org/licenses/by-sa/4.0/.

Attribute the source publisher and preserve applicable share-alike terms when distributing derived data. Those terms do not automatically mean the team's original notebook code and authored report use the same license.

For reproducibility, see the root [README.md](../README.md) and structured [readme.toml](../readme.toml).
