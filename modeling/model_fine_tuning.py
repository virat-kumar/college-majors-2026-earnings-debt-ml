"""Fine-tune the CV-selected model, then open the untouched institution holdout.

Run after model_selection.py: python -m modeling.model_fine_tuning
The validation search uses ONLY training institutions and canonical CV folds.
The final holdout is scored AFTER params are fixed; never used for tuning.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterSampler

from modeling.common import (
    FEATURES, FAMILIES, RANDOM_STATE, ROOT, TARGET,
    folds_for, load_study, pipeline_for, scores,
)

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / ".artifacts"


def candidate_params(winner: str) -> list[dict]:
    original = pipeline_for(winner).named_steps["model"].get_params()
    if winner == "hist_gradient_boosting":
        to_tune = {
            "model__learning_rate": [0.035, 0.055, 0.07, 0.095, 0.12],
            "model__max_iter": [120, 170, 230, 290],
            "model__max_leaf_nodes": [15, 23, 31, 47],
            "model__min_samples_leaf": [15, 30, 50, 75],
            "model__l2_regularization": [0.0, 1.0, 4.0, 10.0],
        }
        original_keys = [
            "learning_rate", "max_iter", "max_leaf_nodes",
            "min_samples_leaf", "l2_regularization"
        ]
        n_samples = 11
    elif winner == "random_forest":
        to_tune = {
            "model__n_estimators": [60, 100, 140],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": [0.6, 0.85, 1.0],
            "model__max_depth": [None, 18, 30],
        }
        original_keys = ["n_estimators", "min_samples_leaf", "max_features", "max_depth"]
        n_samples = 7
    elif winner == "decision_tree":
        to_tune = {
            "model__min_samples_leaf": [10, 20, 30, 50, 75],
            "model__max_depth": [6, 10, 14, 20, None],
            "model__max_features": [0.7, 0.9, None],
        }
        original_keys = ["min_samples_leaf", "max_depth", "max_features"]
        n_samples = 11
    elif winner == "ridge":
        to_tune = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0, 500.0, 2000.0]}
        original_keys = ["alpha"]
        n_samples = 8
    else:
        raise ValueError(f"Unexpected winner {winner!r}")
    baseline = {"model__" + k: original[k] for k in original_keys}
    sampled = list(ParameterSampler(to_tune, n_iter=n_samples, random_state=RANDOM_STATE))
    unique = []
    seen = set()
    for p in [baseline] + sampled:
        signature = json.dumps(p, sort_keys=True, default=str)
        if signature not in seen:
            seen.add(signature)
            unique.append(p)
    return unique


def main() -> None:
    metadata = json.loads((REPORTS / "model_selection.json").read_text(encoding="utf-8"))
    family = metadata["winner"]
    if family not in FAMILIES:
        raise RuntimeError(f"Model selection winner is invalid: {family}")
    train, reserved_test = load_study()
    X_train, y_train = train[FEATURES], train[TARGET]
    folds = folds_for(train)
    candidates = candidate_params(family)
    grids = [{k: [v] for k, v in params.items()} for params in candidates]
    print(f"Tuning {family}: {len(grids)} preselected candidates x 5 canonical "
          f"grouped validation folds (TRAIN only).", flush=True)

    tune_started = time.perf_counter()
    search = GridSearchCV(
        estimator=pipeline_for(family),
        param_grid=grids,
        scoring="neg_mean_absolute_error",
        cv=folds,
        n_jobs=2,
        pre_dispatch=2,
        refit=True,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    elapsed = round(time.perf_counter() - tune_started, 1)
    print(f"Tuning complete: best CV MAE ${-search.best_score_:,.2f} "
          f"parameters {search.best_params_}", flush=True)

    ranked = pd.DataFrame({
        "candidate": np.arange(len(search.cv_results_["params"])),
        "rank": search.cv_results_["rank_test_score"],
        "cv_mae_usd": -search.cv_results_["mean_test_score"],
        "cv_mae_sd_usd": search.cv_results_["std_test_score"],
        "mean_fit_seconds": search.cv_results_["mean_fit_time"],
        "parameters": [json.dumps(p, sort_keys=True) for p in search.cv_results_["params"]],
    }).sort_values(["rank", "candidate"]).reset_index(drop=True)
    ranked.to_csv(REPORTS / "tuning_candidates.csv", index=False, float_format="%.5f")

    # IMPORTANT: held-out test is not used until search.fit/selection has ended.
    X_test, y_test = reserved_test[FEATURES], reserved_test[TARGET]
    tuned_pred = search.best_estimator_.predict(X_test)
    untuned = pipeline_for(family).fit(X_train, y_train)
    untuned_pred = untuned.predict(X_test)
    dummy = pipeline_for("dummy").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    metrics = {
        "selected_family": family,
        "selection_cv_mae_usd": metadata["cv_winner_mae_usd"],
        "original_cv_mae_usd_rechecked_in_tuning": round(float(
            ranked.loc[ranked.candidate == 0, "cv_mae_usd"].iloc[0]
        ), 2),
        "tuned_cv_mae_usd": round(float(-search.best_score_), 2),
        "tuning_search_candidates": len(grids),
        "tuning_grouped_folds": 5,
        "tuning_seconds": elapsed,
        "best_parameters": search.best_params_,
        "train_programs": len(train),
        "test_programs": len(reserved_test),
        "train_institutions": int(train.opeid6.nunique()),
        "test_institutions": int(reserved_test.opeid6.nunique()),
        "test_set_opened_only_after_search": True,
        "holdout": {
            name: {k: round(v, 4) for k, v in scores(y_test, predicted).items()}
            for name, predicted in [
                ("tuned_winner", tuned_pred),
                ("untuned_winner", untuned_pred),
                ("dummy_mean", dummy_pred),
            ]
        },
        "holdout_tuned_absolute_error_p50_usd": round(float(
            np.median(np.abs(y_test.to_numpy() - tuned_pred))
        ), 2),
        "holdout_tuned_absolute_error_p90_usd": round(float(
            np.percentile(np.abs(y_test.to_numpy() - tuned_pred), 90)
        ), 2),
        "methodology": "Group-separated holdout reserved until after fivefold grouped train-only hyperparameter search. Raw USD response; MAE primary. Pipeline refits preprocessing per fold.",
    }
    metrics["holdout_mae_improvement_vs_dummy_percent"] = round(
        100.0 * (metrics["holdout"]["dummy_mean"]["mae_usd"] -
                 metrics["holdout"]["tuned_winner"]["mae_usd"])
        / metrics["holdout"]["dummy_mean"]["mae_usd"], 2
    )
    metrics["holdout_mae_delta_tuned_minus_untuned_usd"] = round(
        metrics["holdout"]["tuned_winner"]["mae_usd"] -
        metrics["holdout"]["untuned_winner"]["mae_usd"], 2
    )
    (REPORTS / "model_evaluation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    # Predictive rows are retained locally (not tracked): release data license
    # and identification of institutions require deliberate publication review.
    ARTIFACTS.mkdir(exist_ok=True)
    predictions = reserved_test[[
        "program_id", "opeid6", "cip_family_title",
        "institution_control", "institution_region",
    ]].copy()
    predictions["earnings_actual_usd"] = y_test.to_numpy()
    predictions["earnings_predicted_usd"] = tuned_pred
    predictions["error_usd"] = tuned_pred - y_test.to_numpy()
    predictions["absolute_error_usd"] = predictions.error_usd.abs()
    predictions.to_csv(ARTIFACTS / "heldout_predictions.csv", index=False)
    joblib.dump(search.best_estimator_, ARTIFACTS / "best_earnings_pipeline.joblib")

    fig, ax = plt.subplots(figsize=(9.2, 5))
    top = ranked.head(10).iloc[::-1]
    labels = [f"Candidate {i}" for i in top.candidate]
    ax.barh(labels, top.cv_mae_usd, color=[
        "#166fa9" if int(v) == int(search.best_index_) else "#91b5cd"
        for v in top.candidate
    ])
    ax.errorbar(top.cv_mae_usd, range(len(top)), xerr=top.cv_mae_sd_usd,
                fmt="none", ecolor="#263648", capsize=3)
    ax.set_xlabel("Grouped CV MAE (USD; lower is better)")
    ax.set_title(f"{family.replace('_', ' ').title()} | best 10 candidate settings")
    ax.grid(axis="x", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "tuning_candidates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.hexbin(y_test.to_numpy(), tuned_pred, gridsize=48, mincnt=1,
              cmap="Blues", bins="log", linewidths=0)
    lower = min(float(np.min(y_test)), float(np.min(tuned_pred)))
    upper = max(float(np.max(y_test)), float(np.max(tuned_pred)))
    ax.plot([lower, upper], [lower, upper], "--", color="#e87436", linewidth=1.6)
    ax.set_xlabel("Actual median earnings four years after completion (USD)")
    ax.set_ylabel("Predicted program-level earnings (USD)")
    ax.set_title("Held-out institutions: predicted versus actual")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    residuals = y_test.to_numpy() - tuned_pred
    lo, hi = np.quantile(residuals, [0.01, 0.99])
    ax.hist(residuals, bins=65, range=(float(lo), float(hi)),
            color="#3981ac", edgecolor="white", alpha=0.91)
    ax.axvline(0, color="#e87436", linestyle="--", linewidth=1.6)
    ax.set_title("Held-out prediction errors (middle 98%; extremes clipped in chart)")
    ax.set_xlabel("Actual earnings minus predicted earnings (USD)")
    ax.set_ylabel("Programs")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_distribution.png", dpi=180)
    plt.close(fig)

    print("HOLDOUT final independent metrics:", flush=True)
    print(json.dumps(metrics["holdout"], indent=2), flush=True)
    print("COMPLETE tuning and one-time holdout evaluation", flush=True)


if __name__ == "__main__":
    main()
