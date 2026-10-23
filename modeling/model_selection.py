"""Compare four model families with the original institution-grouped five folds.

Usage: python -m modeling.model_selection
Evaluates TRAIN ONLY; final holdout is intentionally unseen at this stage.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from modeling.common import (
    FAMILIES, FEATURES, RANDOM_STATE, ROOT, TARGET,
    folds_for, load_study, pipeline_for, scores,
)

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DISPLAY = {
    "dummy": "Mean-only baseline",
    "ridge": "Ridge regression (linear)",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest (bagging)",
    "hist_gradient_boosting": "Histogram gradient boosting",
}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    train, _reserved_test = load_study()
    X, y = train[FEATURES], train[TARGET]
    folds = folds_for(train)
    rows = []
    for family in ("dummy", *FAMILIES):
        for fold, (fit_idx, val_idx) in enumerate(folds):
            estimator = pipeline_for(family)
            t0 = time.perf_counter()
            estimator.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            pred = estimator.predict(X.iloc[val_idx])
            result = scores(y.iloc[val_idx], pred)
            rows.append({
                "model": family, "fold": fold, "fit_n": len(fit_idx),
                "validation_n": len(val_idx),
                "train_institutions": train.iloc[fit_idx].opeid6.nunique(),
                "validation_institutions": train.iloc[val_idx].opeid6.nunique(),
                "fit_and_predict_seconds": round(time.perf_counter() - t0, 3),
                **{key: round(value, 5) for key, value in result.items()},
            })
            print(f"{family:24s} fold {fold} MAE ${result['mae_usd']:,.2f} "
                  f"RMSE ${result['rmse_usd']:,.2f} R2 {result['r2']:.3f}",
                  flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(REPORTS / "model_selection_folds.csv", index=False, float_format="%.5f")
    summaries = []
    for family, part in raw.groupby("model", sort=False):
        summaries.append({
            "model": family,
            "model_family": DISPLAY[family],
            "folds": int(len(part)),
            "cv_mae_usd": round(float(part.mae_usd.mean()), 2),
            "cv_mae_sd_usd": round(float(part.mae_usd.std(ddof=1)), 2),
            "cv_rmse_usd": round(float(part.rmse_usd.mean()), 2),
            "cv_r2": round(float(part.r2.mean()), 4),
            "total_fit_seconds": round(float(part.fit_and_predict_seconds.sum()), 2),
        })
    table = pd.DataFrame(summaries).sort_values("cv_mae_usd").reset_index(drop=True)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)
    contenders = table.loc[table.model != "dummy"]
    winner = str(contenders.iloc[0]["model"])
    baseline = float(table.loc[table.model == "dummy", "cv_mae_usd"].iloc[0])
    winner_mae = float(contenders.iloc[0]["cv_mae_usd"])
    metadata = {
        "target": TARGET, "target_units": "USD", "population": "bachelor programs with reported 4-year earnings",
        "train_programs": len(train), "train_institutions": int(train.opeid6.nunique()),
        "holdout_programs_reserved_and_not_evaluated": len(_reserved_test),
        "cv_method": "Five institution-disjoint fold assignments from data/split_ids.csv",
        "selection_metric": "unweighted mean of five validation-fold MAEs (USD), lower is better",
        "seed": RANDOM_STATE, "predictors": FEATURES,
        "scikit_learn_version": sklearn.__version__,
        "models": {name: pipeline_for(name).named_steps["model"].get_params() for name in ("dummy", *FAMILIES)},
        "winner": winner,
        "cv_winner_mae_usd": winner_mae,
        "cv_mean_baseline_mae_usd": baseline,
        "cv_mae_improvement_vs_dummy_percent": round(100.0 * (baseline - winner_mae) / baseline, 2),
    }
    (REPORTS / "model_selection.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(10, 4.7))
    chart = table.iloc[::-1]
    ys = np.arange(len(chart))
    colors = ["#dededb" if n == "dummy" else ("#1967ac" if n == winner else "#88a6c0")
              for n in chart.model]
    ax.barh(ys, chart.cv_mae_usd, color=colors, height=0.65)
    ax.errorbar(chart.cv_mae_usd, ys, xerr=chart.cv_mae_sd_usd,
                fmt="none", ecolor="#263648", capsize=3, linewidth=1.4)
    ax.set_yticks(ys, chart.model_family)
    ax.set_xlabel("Five-fold cross-validation mean absolute error (USD; lower is better)")
    ax.set_title("Four model families vs. mean-only baseline | bachelor's earnings")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, float(chart.cv_mae_usd.max()) * 1.18)
    for y_coord, metric in zip(ys, chart.cv_mae_usd):
        ax.text(metric + 230, y_coord, f"${metric:,.0f}", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.17)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=180)
    plt.close(fig)

    print("\nCOMPLETE selection", flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"Winner based on grouped CV MAE: {winner} ({winner_mae:,.2f} USD)", flush=True)


if __name__ == "__main__":
    main()
