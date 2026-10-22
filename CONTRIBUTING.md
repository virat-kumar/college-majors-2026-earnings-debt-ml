# Group 2 collaboration guide

## Principles

- Keep the original `College_Majors_EDA_Colab.ipynb` intact. Its Git blob SHA is **`f63ff94c5dd42ffdddb2ad4c7f4cc6b76616994f`**. Put new models in separate notebooks or Python modules.
- Share a common data source, cleaning rules, `opeid6` institution-level split and grouped CV folds.
- Never commit Kaggle datasets, `split_ids.csv`, archives, serialized models, artifacts, notebooks with private data, credentials or virtual environments.
- Prefer small, reviewable changes. Do not reformat the original notebook as part of unrelated work.
- Academic course deliverables require clear attribution and manual review of statistical claims.

## Git workflow

```sh
git switch main
git pull --ff-only
git switch -c feature/your-model
# edit your new files, test them locally
python setup.py --skip-data
git status --short
git add path/to/your/approved/source-files
git commit -m "Add grouped-CV regression model"
git push -u origin feature/your-model
```

Open a pull request to `main`; describe the goal, metrics, data split, target, preprocessing, leakage rules and limitations. At least one teammate should review model methodology before merging.

**Before committing**, run `git status --short`, `git diff --cached --name-only` and confirm no data, secrets or binary outputs are included. The `.gitignore` is a protective default, **not a replacement for review**. GitHub's public visibility means pushed contents can be copied by anyone.

## Agreed and open modeling decisions

The source EDA proposes, but does not finalize:

1. Bachelor's-only earnings regression versus models across credential levels.
2. High-risk debt classification: whether to include it and which cutoff (initially 8%).
3. Final leakage-safe predictor set; whether median debt may be used in earnings regression.
4. Train on raw earnings or log earnings (evaluate dollar-scale errors, with appropriate back-transformation).
5. Common evaluation: held-out institution groups, grouped five-fold CV, MAE baseline and appropriate classification metrics.

Log the final team decision in a pull request or issue, then implement it consistently.

## Pull request checklist

- [ ] Original notebook byte hash remains unchanged.
- [ ] No raw data files or model artifacts are tracked.
- [ ] `python setup.py --skip-data` (or existing venv with `python validate.py`) passes.
- [ ] Sources and credits are cited.
- [ ] Data leakage and institution-group overlap have been checked.
- [ ] All training results include a seed, features, hyperparameters, sample count and evaluation split.
- [ ] Important caveats and suppressed observations are discussed.
