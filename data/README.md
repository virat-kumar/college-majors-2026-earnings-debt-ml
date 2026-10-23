# Reproducible public dataset snapshots

**Version:** [data-v1.0.0](https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml/releases/tag/data-v1.0.0)

This folder contains the versioned, Git-tracked **split_ids.csv** reference for all group members:
- 212,707 program-level rows, 7 columns: program_id, opeid6, credential_level, split, cv_fold, in_regression_set, in_classification_set.
- SHA-256: e5215d887840ae25e0c953793842b1886670f51b879783904988a1348f7bfe6c.

The original raw source CSV is a GitHub Release asset instead of a regular-Git blob:
- [Download college_majors_2026.csv](https://github.com/virat-kumar/college-majors-2026-earnings-debt-ml/releases/download/data-v1.0.0/college_majors_2026.csv).
- 227,980 rows x 72 columns; 126,340,696 bytes.
- SHA-256: 58f1c10e79793f33049f3b381cd81d4d0c190ca1052e427c29779d0c89c5ceca.

## Attribution and license

Source publisher: **kylefengkfeng209**, [College Majors 2026: Earnings, Debt, Jobs, AI](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai).

The Kaggle dataset lists a **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license: https://creativecommons.org/licenses/by-sa/4.0/.

These distributed **CSV datasets** (including the derived split) are shared under the original CC BY-SA 4.0 data license. Attribute the original publisher; share derived data under compatible terms. The team's notebook, reports and Python code retain their respective original authorship, not automatically that data license.

The original notebook writes a *root-level* split_ids.csv during execution, which is ignored. The tracked snapshot is under **data/split_ids.csv**, safe from accidental local overwrite.
