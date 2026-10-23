# Reproducible modeling stages

We preserved the original team EDA notebook exactly and implemented each later phase in a separate Python file. Run from the repository root through the isolated interpreter created by `python setup.py`, or use `python run_project.py`, which selects the correct Windows/Linux interpreter.

1. `eda_audit.py`: independently verify the original cleaning, 212,707-row final population, target coverage and shared-campus decisions; export CSV and JSON audit.
2. `extract_eda_figures.py`: copy all 12 original notebook PNG bytes into `reports/figures/` and store a cell-index/SHA256 manifest; do not change the notebook.
3. `common.py`: read canonical `data/split_ids.csv`; restrict to reported-earnings bachelor's programs; enforce institution-disjoint five-fold training and test; expose 14 feature-specific preprocessing pipelines.
4. `model_selection.py`: compare Ridge, Decision Tree, Random Forest and Histogram Gradient Boosting plus constant mean on the same five training folds, exporting fold CSV, comparison CSV, JSON and figure. **No test outcomes used.**
5. `model_fine_tuning.py`: search the selected family with a deterministic 12-setting training-only grouped search, then evaluate the untouched final holdout **once**, store final metrics, candidates and diagnostic figures. Fitted joblib stays ignored in `.artifacts/`.
6. `model_explainability.py`: evaluate **the frozen model**, with seed-42 permutation importance, segment errors, quintile calibration and 2,000 institution-cluster bootstrap replicates. No post-hoc hyperparameter retuning.

Reviewed numeric results and model explanations are under `reports/`. The submission-ready PDF and its source Markdown are in `submission/`. See `tests/` for integrity checks.
