"""Shared, leakage-audited BUAN 6341 bachelor's earnings experiment design.

All model selection happens on the original EDA notebook's institution-grouped
training partition, with preprocessors fitted independently inside each fold.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TARGET = "median_earnings_4yr_usd"
NUMERIC = (
    "institution_admission_rate", "institution_avg_sat",
    "institution_undergrad_enrollment", "institution_tuition_in_state_usd",
    "institution_tuition_out_state_usd", "awards_year1",
    "occupation_growth_pct_2024_34", "occupation_median_wage_2024_usd",
    "ai_software_occupation_share",
)
BINARY = ("institution_is_hbcu",)
CATEGORICAL = (
    "institution_control", "cip_family_title", "institution_region",
    "distance_education",
)
FEATURES = list(NUMERIC + BINARY + CATEGORICAL)
FAMILIES = ("ridge", "decision_tree", "random_forest", "hist_gradient_boosting")
EXPECTED_NOTEBOOK_SHA256 = "418649971708968b7d999cfc9dc08fc09a9890f331c0ab32ac48829dfbdafe42"


def load_study() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test *only for reported-earnings bachelor's programs*.

    The canonical split supplies train/test and CV membership, never a new
    random split. The held-out target is loaded but not used by selection code.
    """
    raw_path = ROOT / "college_majors_2026.csv"
    split_path = ROOT / "data" / "split_ids.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Run 'python setup.py' before modeling to download the verified dataset.")
    required = ["program_id", "opeid6", "credential_level", TARGET] + FEATURES
    raw = pd.read_csv(raw_path, usecols=required, low_memory=False)
    split = pd.read_csv(split_path)
    if raw.program_id.duplicated().any() or split.program_id.duplicated().any():
        raise AssertionError("program_id must uniquely identify programs.")
    full = split.merge(
        raw, on="program_id", how="left", validate="one_to_one",
        indicator=True, suffixes=("_reference", ""),
    )
    if not (full["_merge"] == "both").all() or len(full) != len(split):
        raise AssertionError("All canonical split IDs must match one original CSV row.")
    for col in ("opeid6", "credential_level"):
        if not full[f"{col}_reference"].equals(full[col]):
            raise AssertionError(f"{col} differs between the release and canonical split.")
        full = full.drop(columns=[f"{col}_reference"])
    full = full.drop(columns=["_merge"])
    if set(full.split.unique()) != {"train", "test"}:
        raise AssertionError("Only the original train/test partitions are allowed.")
    institutional_folds = full.loc[full.split == "train"].groupby("opeid6")["cv_fold"].nunique()
    if (institutional_folds != 1).any():
        raise AssertionError("An institution appears in multiple training CV folds.")
    if not set(full.loc[full.split == "train", "opeid6"]).isdisjoint(
        full.loc[full.split == "test", "opeid6"]
    ):
        raise AssertionError("Institution leakage between training and test!")
    if set(full.loc[full.split == "train", "cv_fold"]) != set(range(5)):
        raise AssertionError("Expected five canonical group-based CV folds.")
    if not (full.loc[full.split == "test", "cv_fold"] == -1).all():
        raise AssertionError("Canonical test rows should have cv_fold=-1.")

    study = full.loc[
        (full.credential_level == 3) & full.in_regression_set
    ].copy()
    if study[TARGET].isna().any() or (study[TARGET] <= 0).any():
        raise AssertionError("Study requires reported positive earnings targets.")
    train = study.loc[study.split == "train"].copy().reset_index(drop=True)
    test = study.loc[study.split == "test"].copy().reset_index(drop=True)
    if train.empty or test.empty or set(train.cv_fold) != set(range(5)):
        raise AssertionError("Incomplete bachelor's earning study partitions.")
    if not set(train.opeid6).isdisjoint(test.opeid6):
        raise AssertionError("Institution leakage among eligible programs.")
    return train, test


def folds_for(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Five institution-disjoint folds *as written by the original notebook*."""
    folds = []
    for k in range(5):
        indices_train = np.flatnonzero((train.cv_fold != k).to_numpy())
        indices_valid = np.flatnonzero((train.cv_fold == k).to_numpy())
        assert len(indices_train) > 0 and len(indices_valid) > 0
        assert set(train.iloc[indices_train].opeid6).isdisjoint(
            set(train.iloc[indices_valid].opeid6)
        ), "CV fold leaked an institution"
        folds.append((indices_train, indices_valid))
    return folds


def preprocessing(scale: bool) -> ColumnTransformer:
    numeric_steps = [
        ("median_and_missing_flags", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), list(NUMERIC)),
            ("binary", SimpleImputer(strategy="most_frequent"), list(BINARY)),
            ("categorical", Pipeline([
                ("missing", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("one_hot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                )),
            ]), list(CATEGORICAL)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def model_for(name: str):
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(alpha=50.0)
    if name == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=14, min_samples_leaf=30, random_state=RANDOM_STATE
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=70, min_samples_leaf=5, max_features=0.85,
            random_state=RANDOM_STATE, n_jobs=2
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=170, learning_rate=0.07, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=False, random_state=RANDOM_STATE
        )
    raise ValueError(f"Unknown model: {name}")


def pipeline_for(name: str) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessing(scale=(name == "ridge"))),
        ("model", model_for(name)),
    ])


def scores(y_true, y_pred) -> dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae_usd": float(mean_absolute_error(y_true, y_pred)),
        "rmse_usd": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
