from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.selectors.cmim import CMIMSelector
from src.selectors.mrmr import MRMRSelector
from src.selectors.proxy_rank import ProxyRankSelector
from src.selectors.fair_mrmr import FairMRMRSelector
from src.selectors.fair_cfs import FairCFSStyleSelector
from src.selectors.fair_lasso import FairLassoStyleSelector
from src.selectors.fa_cmim_basic import FACMIMBasicSelector
from src.selectors.fa_cmim_subset import FACMIMSubsetAwareSelector


ACS_RENAME_MAP = {
    "AGEP": "age",
    "COW": "class-worker",
    "SCHL": "education",
    "MAR": "marital-status",
    "OCCP": "occupation",
    "POBP": "birthplace",
    "RELP": "relationship",
    "WKHP": "hours-per-week",
    "SEX": "sex",
    "RAC1P": "race",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def make_synthetic_acs_income_smoke(
    n: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)

    sex = rng.integers(0, 2, size=n)
    race = rng.integers(1, 5, size=n)

    age = rng.integers(18, 75, size=n)
    education = rng.integers(1, 24, size=n)
    hours = np.clip(rng.normal(40 + 4 * sex, 10, size=n).round(), 1, 80).astype(int)

    marital_status = np.where(
        rng.random(n) < np.where(sex == 1, 0.48, 0.38),
        "married",
        "not-married",
    )

    relationship = np.where(
        sex == 1,
        rng.choice(["husband", "not-in-family", "own-child"], size=n, p=[0.42, 0.45, 0.13]),
        rng.choice(["wife", "not-in-family", "own-child"], size=n, p=[0.28, 0.55, 0.17]),
    )

    occupation = rng.choice(
        ["professional", "service", "admin", "sales", "manual"],
        size=n,
        p=[0.25, 0.20, 0.20, 0.20, 0.15],
    )

    class_worker = rng.choice(
        ["private", "government", "self-employed"],
        size=n,
        p=[0.72, 0.18, 0.10],
    )

    birthplace = rng.choice(
        ["US", "non-US"],
        size=n,
        p=[0.85, 0.15],
    )

    proxy_bonus = (
        (marital_status == "married").astype(float)
        + (relationship == "husband").astype(float)
        + 0.5 * (occupation == "professional").astype(float)
    )

    latent_score = (
        0.045 * (age - 35)
        + 0.22 * (education - 10)
        + 0.035 * (hours - 40)
        + 0.85 * proxy_bonus
        + 0.35 * sex
        + rng.normal(0, 1.5, size=n)
    )

    y = (latent_score > np.quantile(latent_score, 0.62)).astype(int)

    X = pd.DataFrame(
        {
            "age": age,
            "class-worker": class_worker,
            "education": education,
            "marital-status": marital_status,
            "occupation": occupation,
            "birthplace": birthplace,
            "relationship": relationship,
            "hours-per-week": hours,
            "sex": sex,
            "race": race,
        }
    )

    y = pd.Series(y, name="income_gt_50k")
    sensitive = pd.Series(sex, name="sex")

    return X, y, sensitive


def load_acs_income(
    *,
    sample_size: int,
    random_state: int,
    state: str,
    survey_year: str,
    sensitive_col: str,
    synthetic_smoke: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    if synthetic_smoke:
        X, y, sensitive = make_synthetic_acs_income_smoke(
            n=sample_size,
            random_state=random_state,
        )
        metadata = {
            "dataset": "ACSIncome synthetic smoke",
            "sample_size": int(len(X)),
            "sensitive_col": sensitive_col,
            "state": None,
            "survey_year": None,
        }
        return X, y, sensitive, metadata

    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError as exc:
        raise ImportError(
            "folktables is required for real ACSIncome runs. "
            "Install it in Colab with: !pip install folktables==0.0.12"
        ) from exc

    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon="1-Year",
        survey="person",
    )

    acs_data = data_source.get_data(states=[state], download=True)

    X_np, y_np, _ = ACSIncome.df_to_numpy(acs_data)

    feature_names = list(ACSIncome.features)
    X = pd.DataFrame(X_np, columns=feature_names)
    X = X.rename(columns=ACS_RENAME_MAP)

    y = pd.Series(y_np.astype(int), name="income_gt_50k")

    if sample_size is not None and sample_size > 0 and sample_size < len(X):
        sampled = X.copy()
        sampled["__target__"] = y.to_numpy()
        sampled = sampled.sample(n=sample_size, random_state=random_state)

        y = sampled["__target__"].astype(int).reset_index(drop=True)
        X = sampled.drop(columns=["__target__"]).reset_index(drop=True)

    if sensitive_col not in X.columns:
        raise ValueError(
            f"Sensitive column {sensitive_col!r} not found. "
            f"Available columns: {list(X.columns)}"
        )

    sensitive = X[sensitive_col].copy()
    sensitive.name = sensitive_col

    metadata = {
        "dataset": "ACSIncome",
        "sample_size": int(len(X)),
        "state": state,
        "survey_year": survey_year,
        "sensitive_col": sensitive_col,
        "feature_columns": list(X.columns),
        "target_name": y.name,
    }

    return X, y, sensitive, metadata


def _make_selector(
    cls,
    *,
    k: int,
    random_state: int,
    fairness_penalty: float | None = None,
):
    kwargs: dict[str, Any] = {"k": k}

    if fairness_penalty is not None:
        kwargs["fairness_penalty"] = fairness_penalty

    try:
        return cls(**kwargs, random_state=random_state)
    except TypeError:
        return cls(**kwargs)


def build_selectors(
    *,
    k: int,
    lambdas: list[float],
    random_state: int,
):
    selectors = [
        (f"CMIM_k{k}", _make_selector(CMIMSelector, k=k, random_state=random_state)),
        (f"mRMR_k{k}", _make_selector(MRMRSelector, k=k, random_state=random_state)),
    ]

    for lam in lambdas:
        selectors.extend(
            [
                (
                    f"ProxyRank_k{k}_lambda{lam}",
                    _make_selector(
                        ProxyRankSelector,
                        k=k,
                        random_state=random_state,
                        fairness_penalty=lam,
                    ),
                ),
                (
                    f"FairmRMR_k{k}_lambda{lam}",
                    _make_selector(
                        FairMRMRSelector,
                        k=k,
                        random_state=random_state,
                        fairness_penalty=lam,
                    ),
                ),
                (
                    f"FairCFS_k{k}_lambda{lam}",
                    _make_selector(
                        FairCFSStyleSelector,
                        k=k,
                        random_state=random_state,
                        fairness_penalty=lam,
                    ),
                ),
                (
                    f"FairLasso_k{k}_lambda{lam}",
                    _make_selector(
                        FairLassoStyleSelector,
                        k=k,
                        random_state=random_state,
                        fairness_penalty=lam,
                    ),
                ),
                (
                    f"BasicFACMIM_k{k}_lambda{lam}",
                    _make_selector(
                        FACMIMBasicSelector,
                        k=k,
                        random_state=random_state,
                        fairness_penalty=lam,
                    ),
                ),
                (
                    f"SubsetFACMIM_k{k}_lambda{lam}",
                    _make_selector(
                        FACMIMSubsetAwareSelector,
                        k=k,
                        random_state=random_state,
                        fairness_penalty=lam,
                    ),
                ),
            ]
        )

    return selectors


def _fit_selector(selector, X_train, y_train, sensitive_train):
    try:
        selector.fit(X_train, y_train, sensitive=sensitive_train)
    except TypeError:
        selector.fit(X_train, y_train, sensitive_train)

    if hasattr(selector, "get_selected_features"):
        selected = selector.get_selected_features()
    elif hasattr(selector, "get_support"):
        selected = selector.get_support()
    else:
        selected = list(selector.selected_features_)

    selected = [c for c in selected if c in X_train.columns]

    if not selected:
        raise RuntimeError(f"Selector {selector} selected no usable features.")

    return selector, selected


def _transform_selected(selector, X, selected):
    if hasattr(selector, "transform"):
        try:
            Xt = selector.transform(X)
            if isinstance(Xt, pd.DataFrame):
                return Xt
        except Exception:
            pass

    return X[selected].copy()


def _encode_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            median = pd.to_numeric(X_train[col], errors="coerce").median()
            if pd.isna(median):
                median = 0.0
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(median)
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(median)
        else:
            X_train[col] = X_train[col].astype("object").where(X_train[col].notna(), "__missing__").astype(str)
            X_test[col] = X_test[col].astype("object").where(X_test[col].notna(), "__missing__").astype(str)

    X_train_enc = pd.get_dummies(X_train, dummy_na=False)
    X_test_enc = pd.get_dummies(X_test, dummy_na=False)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    return X_train_scaled, X_test_scaled


def _make_model(model_name: str, random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)

    raise ValueError(f"Unknown model: {model_name}")


def _positive_rate_gap(y_pred, sensitive):
    rates = []

    for group in pd.Series(sensitive).dropna().unique():
        mask = np.asarray(sensitive) == group
        if mask.sum() == 0:
            continue
        rates.append(float(np.mean(np.asarray(y_pred)[mask] == 1)))

    if len(rates) < 2:
        return 0.0, 1.0

    min_rate = min(rates)
    max_rate = max(rates)

    dpd = max_rate - min_rate
    dpr = min_rate / max_rate if max_rate > 0 else 1.0

    return float(dpd), float(dpr)


def _rate_gap_by_group(y_true, y_pred, sensitive, positive_label: int):
    rates = []

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    for group in pd.Series(sensitive).dropna().unique():
        group_mask = sensitive == group
        denom_mask = group_mask & (y_true == positive_label)

        if denom_mask.sum() == 0:
            continue

        if positive_label == 1:
            rate = np.mean(y_pred[denom_mask] == 1)
        else:
            rate = np.mean(y_pred[denom_mask] == 1)

        rates.append(float(rate))

    if len(rates) < 2:
        return 0.0

    return float(max(rates) - min(rates))


def _fairness_metrics(y_true, y_pred, sensitive):
    dpd, dpr = _positive_rate_gap(y_pred, sensitive)

    tpr_gap = _rate_gap_by_group(y_true, y_pred, sensitive, positive_label=1)
    fpr_gap = _rate_gap_by_group(y_true, y_pred, sensitive, positive_label=0)

    return {
        "dpd": dpd,
        "dpr": dpr,
        "equal_opportunity_difference": tpr_gap,
        "equalized_odds_difference": max(tpr_gap, fpr_gap),
    }


def _discretize_for_mi(s: pd.Series, n_bins: int = 5) -> pd.Series:
    s = pd.Series(s)

    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors="coerce")

        if numeric.nunique(dropna=True) <= n_bins:
            return numeric.astype("object").where(numeric.notna(), "__missing__").astype(str)

        try:
            return pd.qcut(
                numeric.rank(method="first"),
                q=n_bins,
                duplicates="drop",
            ).astype(str)
        except Exception:
            return numeric.astype("object").where(numeric.notna(), "__missing__").astype(str)

    return s.astype("object").where(s.notna(), "__missing__").astype(str)


def _joint_encode(X: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    if X.empty:
        return pd.Series(["__empty__"] * len(X), index=X.index)

    parts = []

    for col in X.columns:
        parts.append(_discretize_for_mi(X[col], n_bins=n_bins).astype(str))

    joint = pd.concat(parts, axis=1).agg("|".join, axis=1)

    return joint


def _proxy_leakage_metrics(X_selected: pd.DataFrame, sensitive: pd.Series):
    sensitive_d = _discretize_for_mi(sensitive)

    mi_values = []

    for col in X_selected.columns:
        feature_d = _discretize_for_mi(X_selected[col])
        mi_values.append(float(mutual_info_score(feature_d, sensitive_d)))

    mean_mi = float(np.mean(mi_values)) if mi_values else 0.0

    joint_state = _joint_encode(X_selected)
    joint_mi = float(mutual_info_score(joint_state, sensitive_d))

    return mean_mi, joint_mi


def _sensitive_attacker_balanced_accuracy(
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
    sensitive_train: pd.Series,
    sensitive_test: pd.Series,
):
    sensitive_train_codes = pd.Series(sensitive_train).astype("category").cat.codes
    sensitive_test_codes = pd.Series(sensitive_test).astype("category").cat.codes

    if sensitive_train_codes.nunique() < 2 or sensitive_test_codes.nunique() < 2:
        return np.nan

    X_train_enc, X_test_enc = _encode_train_test(X_train_selected, X_test_selected)

    attacker = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )

    try:
        attacker.fit(X_train_enc, sensitive_train_codes)
        pred = attacker.predict(X_test_enc)
        return float(balanced_accuracy_score(sensitive_test_codes, pred))
    except Exception:
        return np.nan


def evaluate_model(
    *,
    model_name: str,
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sensitive_train: pd.Series,
    sensitive_test: pd.Series,
    random_state: int,
):
    X_train_enc, X_test_enc = _encode_train_test(X_train_selected, X_test_selected)

    model = _make_model(model_name, random_state=random_state)
    model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_enc)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_enc)
    else:
        y_score = y_pred

    try:
        auroc = float(roc_auc_score(y_test, y_score))
    except Exception:
        auroc = np.nan

    fairness = _fairness_metrics(y_test, y_pred, sensitive_test)

    mean_mi, joint_mi = _proxy_leakage_metrics(X_test_selected, sensitive_test)

    attacker_ba = _sensitive_attacker_balanced_accuracy(
        X_train_selected,
        X_test_selected,
        sensitive_train,
        sensitive_test,
    )

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": auroc,
        **fairness,
        "mean_selected_mi_sensitive": mean_mi,
        "joint_subset_mi_sensitive": joint_mi,
        "sensitive_attacker_balanced_accuracy": attacker_ba,
    }


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    X, y, sensitive, dataset_metadata = load_acs_income(
        sample_size=args.sample_size,
        random_state=args.random_state,
        state=args.state,
        survey_year=args.survey_year,
        sensitive_col=args.sensitive_col,
        synthetic_smoke=args.synthetic_smoke,
    )

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    selectors = build_selectors(
        k=args.k,
        lambdas=args.lambdas,
        random_state=args.random_state,
    )

    rows = []

    for selector_name, selector in selectors:
        print(f"\nSelector: {selector_name}")
        selector_start = time.time()

        fitted_selector, selected_features = _fit_selector(
            selector,
            X_train,
            y_train,
            sensitive_train,
        )

        X_train_selected = _transform_selected(fitted_selector, X_train, selected_features)
        X_test_selected = _transform_selected(fitted_selector, X_test, selected_features)

        selector_runtime = time.time() - selector_start

        if hasattr(fitted_selector, "get_diagnostics"):
            try:
                diag = fitted_selector.get_diagnostics()
                diag_path = diagnostics_dir / f"{_safe_name(selector_name)}_diagnostics.csv"
                diag.to_csv(diag_path, index=False)
            except Exception as exc:
                print(f"Warning: could not save diagnostics for {selector_name}: {exc}")

        for model_name in args.models:
            print(f"  Model: {model_name}")

            metrics = evaluate_model(
                model_name=model_name,
                X_train_selected=X_train_selected,
                X_test_selected=X_test_selected,
                y_train=y_train,
                y_test=y_test,
                sensitive_train=sensitive_train,
                sensitive_test=sensitive_test,
                random_state=args.random_state,
            )

            rows.append(
                {
                    "dataset": "ACSIncome",
                    "selector": selector_name,
                    **metrics,
                    "k": args.k,
                    "lambda_values": "|".join(str(x) for x in args.lambdas),
                    "selector_runtime_seconds": selector_runtime,
                    "n_selected_features": len(selected_features),
                    "selected_features": "|".join(selected_features),
                }
            )

    results = pd.DataFrame(rows)

    results_path = output_dir / "acs_income_pilot_results.csv"
    results.to_csv(results_path, index=False)

    metadata = {
        **dataset_metadata,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "k": args.k,
        "lambdas": args.lambdas,
        "models": args.models,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_result_rows": int(len(results)),
        "output_dir": str(output_dir),
    }

    metadata_path = output_dir / "acs_income_pilot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved results to: {results_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Rows: {len(results)}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACSIncome pilot experiment.")

    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "gradient_boosting"],
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--output_dir", type=str, default="results/acs_income/final")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.3)

    parser.add_argument("--state", type=str, default="CA")
    parser.add_argument("--survey_year", type=str, default="2018")
    parser.add_argument("--sensitive_col", type=str, default="sex")

    parser.add_argument(
        "--synthetic_smoke",
        action="store_true",
        help="Use synthetic ACS-like data instead of downloading Folktables ACSIncome.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
