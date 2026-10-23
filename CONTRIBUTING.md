# Group 2 collaboration guide

## Principles

- Keep the original `College_Majors_EDA_Colab.ipynb` intact. Its Git blob SHA is **`f63ff94c5dd42ffdddb2ad4c7f4cc6b76616994f`**. Put new models in separate notebooks or Python modules.
- Share a common data source, cleaning rules, `opeid6` institution-level split and grouped CV folds.
- The reviewed CSV snapshots are an explicit exception: the raw source is a GitHub Release asset and the canonical grouped split is tracked at `data/split_ids.csv` under CC BY-SA 4.0. The completed project also explicitly tracks reviewed `reports/*.csv` model/audit artifacts, original EDA figure copies, and the finished `submission/*.pdf` report. Never commit arbitrary additional datasets, generated root-level `split_ids.csv`, archives, private models, secrets or environments without review.
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

Open a pull request to `main`; describe the goal, metrics, data split, target, preprocessing, leakage rules and limitations. Teammates should review methodology, hashes and source attribution when possible; the finished report explicitly states what remains unvalidated for operational use.

**Before committing**, run `git status --short`, `git diff --cached --name-only` and confirm only the explicitly reviewed public datasets (`data/split_ids.csv`, original CSV via GitHub Release), model/audit `reports/*.csv`, project scripts and docs, approved figures and submission PDF are included; no secrets, environments, unintended binaries or other datasets. The `.gitignore` is a protective default, **not a replacement for review**. GitHub's public visibility means pushed contents can be copied by anyone.

## Modeling decisions and completed experiments

The project has completed **bachelor's-only four-year reported earnings regression** using the original `opeid6` institution holdout and canonical `cv_fold` labels. We used the original 14-feature proposal without debt-derived predictors, fitted all preprocessing within each training fold, and modeled raw USD earnings (log-earnings was explored only in EDA). We compared Ridge, Decision Tree, Random Forest and Histogram Gradient Boosting; selected boosting by grouped CV MAE; tested 12 hyperparameter configurations; and used the 379-school holdout only for final evaluation.

High debt-risk is an **EDA-only exploratory task**, not a trained classifier in this deliverable. A future classifier would need a separate target, leakage audit and group-approved threshold. Completed evidence and the direct submission PDF are linked in `README.md`.

## Pull request checklist

- [ ] Original notebook byte hash remains unchanged.
- [ ] Only reviewed `data/split_ids.csv`, `reports/*.csv`, approved figures and final PDF are tracked; the raw large CSV remains in the versioned public GitHub Release.
- [ ] `python setup.py --skip-data` (or existing venv with `python validate.py`) passes.
- [ ] Sources and credits are cited.
- [ ] Data leakage and institution-group overlap have been checked.
- [ ] All training results include a seed, features, hyperparameters, sample count and evaluation split.
- [ ] Important caveats and suppressed observations are discussed.
