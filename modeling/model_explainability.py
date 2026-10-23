"""Explain the frozen tuned predictor; NO feature selection or retuning.

Run once after model_fine_tuning.py:
python -m modeling.model_explainability

Permutation importance is computed on a reproducible 2,000-row subset of
held-out programs. The test set is for *post-selection diagnosis only*.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from modeling.common import FEATURES, ROOT, TARGET, load_study, pipeline_for, scores

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / ".artifacts"


def institution_bootstrap(
    actual: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray,
    n_resamples: int = 2000, seed: int = 42
) -> dict:
    """95% bootstrap CI for MAE(a) - MAE(b), resampling held-out institutions."""
    frame = pd.DataFrame({
        "group": groups, "error_a": np.abs(a - actual),
        "error_b": np.abs(b - actual),
    }).groupby("group", sort=True).agg(
        error_a=("error_a", "sum"), error_b=("error_b", "sum"), rows=("error_a", "size")
    )
    rng = np.random.default_rng(seed)
    grouped = frame[["error_a", "error_b", "rows"]].to_numpy(dtype=float)
    n_groups = len(grouped)
    indices = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    sampled = grouped[indices].sum(axis=1)
    differences = (sampled[:, 0] - sampled[:, 1]) / sampled[:, 2]
    point = float(np.mean(np.abs(a - actual)) - np.mean(np.abs(b - actual)))
    return {
        "point_delta_mae_a_minus_b_usd": round(point, 2),
        "lower_95_usd": round(float(np.quantile(differences, .025)), 2),
        "upper_95_usd": round(float(np.quantile(differences, .975)), 2),
        "n_institutions": n_groups,
        "bootstrap_repeats": n_resamples,
        "seed": seed,
        "interpretation": "Paired institution-cluster bootstrap; descriptive, not a formal model-superiority test.",
    }


def main() -> None:
    train, test = load_study()
    metadata = json.loads((REPORTS / "model_evaluation.json").read_text(encoding="utf-8"))
    final = joblib.load(ARTIFACTS / "best_earnings_pipeline.joblib")
    X, y = test[FEATURES], test[TARGET].to_numpy(dtype=float)
    pred = final.predict(X)
    # Independently reproduce the pre-tuning control and mean-only baseline.
    untuned = pipeline_for(metadata["selected_family"])
    untuned.fit(train[FEATURES], train[TARGET])
    original_pred = untuned.predict(X)
    mean_pred = np.full_like(y, float(train[TARGET].mean()))
    assert abs(scores(y, pred)["mae_usd"] - metadata["holdout"]["tuned_winner"]["mae_usd"]) < .0002
    assert abs(scores(y, original_pred)["mae_usd"] - metadata["holdout"]["untuned_winner"]["mae_usd"]) < .0002
    assert abs(scores(y, mean_pred)["mae_usd"] - metadata["holdout"]["dummy_mean"]["mae_usd"]) < .0002

    comparison = institution_bootstrap(y, pred, original_pred, test["opeid6"].to_numpy())
    vs_baseline = institution_bootstrap(y, pred, mean_pred, test["opeid6"].to_numpy())
    print("PAIRED institution bootstrap tuned - untuned:",comparison,flush=True)
    print("PAIRED institution bootstrap tuned - dummy:",vs_baseline,flush=True)

    # Full independent held-out predictions are public, institute/program aggregates,
    # already represented by the public Kaggle dataset, and contain no student PII.
    records = test[[
        "program_id", "opeid6", "credential_level", "institution_control",
        "institution_region", "cip_family_title",
    ]].copy()
    records["actual_earnings_4yr_usd"] = y.round(2)
    records["tuned_prediction_usd"] = pred.round(2)
    records["original_prediction_usd"] = original_pred.round(2)
    records["absolute_error_usd"] = np.abs(y-pred).round(2)
    records["error_actual_minus_prediction_usd"] = (y-pred).round(2)
    records.to_csv(REPORTS / "heldout_predictions.csv", index=False)

    slices = []
    for col, minimum in [
        ("cip_family_title", 50),
        ("institution_control", 50),
        ("institution_region", 50),
    ]:
        frame = records.dropna(subset=[col]).copy()
        for value, part in frame.groupby(col, sort=True):
            if len(part) >= minimum:
                slices.append({
                    "dimension": col, "segment": value, "n_programs": len(part),
                    "n_institutions": part.opeid6.nunique(),
                    "median_actual_earnings_usd": round(float(part.actual_earnings_4yr_usd.median()), 2),
                    "mean_absolute_error_usd": round(float(part.absolute_error_usd.mean()), 2),
                    "mean_signed_error_usd": round(float(part.error_actual_minus_prediction_usd.mean()), 2),
                })
    slice_df = pd.DataFrame(slices).sort_values(["dimension", "n_programs"], ascending=[True, False])
    slice_df.to_csv(REPORTS / "error_by_segment.csv", index=False)
    print("ERROR SLICES",len(slice_df),flush=True)

    # Permuting RAW inputs rather than transformed OHE columns yields feature-level
    # MAE loss in USD. These importances describe associations, not causality.
    rng = np.random.default_rng(42)
    sampled = np.sort(rng.choice(len(test), size=min(2000, len(test)), replace=False))
    result = permutation_importance(
        final, X.iloc[sampled], y[sampled], scoring="neg_mean_absolute_error",
        n_repeats=3, n_jobs=2, random_state=42
    )
    fi = pd.DataFrame({
        "feature": FEATURES,
        "delta_mae_usd": np.round(result.importances_mean, 2),
        "delta_mae_sd_usd": np.round(result.importances_std, 2),
        "n_eval_rows": len(sampled),
        "n_repeats": 3,
    }).sort_values("delta_mae_usd", ascending=False)
    fi.to_csv(REPORTS / "feature_importance.csv", index=False)
    print("PERMUTATION IMPORTANCE",fi.head(9).to_string(index=False),flush=True)

    FIGURES.mkdir(parents=True, exist_ok=True)
    top = fi.head(11).iloc[::-1]
    fig,ax=plt.subplots(figsize=(9.8,5.5))
    ax.barh(range(len(top)),top.delta_mae_usd,color="#276b9e")
    ax.errorbar(top.delta_mae_usd,range(len(top)),xerr=top.delta_mae_sd_usd,
                fmt="none",ecolor="#1d3142",capsize=2)
    ax.set_yticks(range(len(top)),[
        s.replace("institution_","").replace("occupation_","occ_").replace("_"," ")
        for s in top.feature
    ])
    ax.set_title("Permutation importance | locked final model, 2,000 held-out programs")
    ax.set_xlabel("Increase in MAE when shuffled (USD; higher = more predictive)")
    ax.grid(axis="x",alpha=.17);ax.spines[["right","top"]].set_visible(False)
    fig.tight_layout();fig.savefig(FIGURES/"permutation_importance.png",dpi=185);plt.close(fig)

    g=slice_df[(slice_df.dimension=="cip_family_title")&(slice_df.n_programs>=90)].sort_values(
        "n_programs",ascending=False
    ).head(12).sort_values("mean_absolute_error_usd")
    fig,ax=plt.subplots(figsize=(9.3,5.5))
    ys=np.arange(len(g))
    ax.barh(ys,g.mean_absolute_error_usd,color="#3885ad")
    ax.set_yticks(ys,[
        (str(label)[:31]+"..." if len(str(label))>34 else str(label))
        for label in g.segment
    ])
    for j, (_,rr) in enumerate(g.iterrows()):
        ax.text(float(rr.mean_absolute_error_usd)+150,j,f"n={int(rr.n_programs):,}",va="center",fontsize=8)
    ax.set_xlim(0,float(g.mean_absolute_error_usd.max())*1.32)
    ax.set_title("Errors differ by major family | most common held-out fields")
    ax.set_xlabel("Held-out MAE (USD; unadjusted for field mix)")
    ax.grid(axis="x",alpha=.14);ax.spines[["right","top"]].set_visible(False)
    fig.tight_layout();fig.savefig(FIGURES/"errors_by_major.png",dpi=185);plt.close(fig)

    # Actual outcomes are only inspected here for descriptive diagnostics,
    # never for post-hoc feature or hyperparameter selection.
    q=pd.qcut(records["actual_earnings_4yr_usd"],5,labels=False,duplicates="drop")
    qtable=records.groupby(q,observed=True).agg(
        n=("program_id","size"),
        actual_mean_usd=("actual_earnings_4yr_usd","mean"),
        prediction_mean_usd=("tuned_prediction_usd","mean"),
        mae_usd=("absolute_error_usd","mean"),
    ).reset_index().rename(columns={"actual_earnings_4yr_usd":"actual_earnings_quintile"})
    qtable.to_csv(REPORTS/"error_by_actual_earnings_quintile.csv",index=False,float_format="%.2f")
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    xx=np.arange(len(qtable));w=.35
    ax.bar(xx-w/2,qtable.actual_mean_usd,width=w,label="Actual mean",color="#2f719d")
    ax.bar(xx+w/2,qtable.prediction_mean_usd,width=w,label="Predicted mean",color="#e5a05c")
    ax.set_xticks(xx,[f"Q{int(v)+1}" for v in qtable.iloc[:,0]])
    ax.set_ylabel("Mean program earnings (USD)")
    ax.set_xlabel("Quintile of observed held-out earnings")
    ax.set_title("Error concentration across low- and high-earning programs")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.16);ax.spines[["right","top"]].set_visible(False)
    fig.tight_layout();fig.savefig(FIGURES/"earnings_quintile_calibration.png",dpi=185);plt.close(fig)

    summary={
        "permutation_eval_rows":len(sampled),
        "permutation_repeats":3,
        "permutation_scoring":"reduction in negative MAE / loss of prediction accuracy, USD",
        "bootstrap_tuned_minus_untuned":comparison,
        "bootstrap_tuned_minus_dummy":vs_baseline,
        "top_features":[{"feature":row.feature,"delta_mae_usd":float(row.delta_mae_usd)}
                       for row in fi.head(8).itertuples()],
        "slice_threshold":50,
        "test_n":len(test),
        "diagnostic_data_use":"Post-model-freeze heldout analysis only; no hyperparameter or feature changes.",
        "caution":"Error heterogeneity is descriptive; correlated predictors and population suppression limit causal interpretation.",
    }
    (REPORTS/"explainability_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("EXPLAINABILITY COMPLETE",flush=True)


if __name__=="__main__":
    main()
