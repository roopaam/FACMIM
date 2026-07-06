from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from scripts.run_acs_income_pilot import (
    _fit_selector,
    _safe_name,
    _transform_selected,
    build_selectors,
    evaluate_model,
)


DEFAULT_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/"
    "compas-analysis/master/compas-scores-two-years.csv"
)

DEFAULT_COMPAS_FEATURES = [
    "age",
    "age_cat",
    "sex",
    "race",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
    "c_charge_desc",
]


def make_synthetic_compas_smoke(
    n: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)

    race = rng.choice(
        ["African-American", "Caucasian", "Hispanic", "Other"],
        size=n,
        p=[0.47, 0.36, 0.10, 0.07],
    )

    sex = rng.choice(["Male", "Female"], size=n, p=[0.78, 0.22])

    age = rng.integers(18, 70, size=n)

    age_cat = pd.cut(
        age,
        bins=[0, 25, 45, 120],
        labels=["Less than 25", "25 - 45", "Greater than 45"],
        include_lowest=True,
    ).astype(str)

    priors_count = rng.poisson(
        lam=np.where(race == "African-American", 3.0, 2.0),
        size=n,
    )

    priors_count = np.clip(priors_count, 0, 20)

    juv_fel_count = rng.binomial(2, 0.08, size=n)
    juv_misd_count = rng.binomial(3, 0.10, size=n)
    juv_other_count = rng.binomial(3, 0.10, size=n)

    c_charge_degree = rng.choice(["F", "M"], size=n, p=[0.63, 0.37])

    c_charge_desc = rng.choice(
        [
            "Battery",
            "Theft",
            "Drug Possession",
            "Burglary",
            "Assault",
            "Driving License",
            "Other",
        ],
        size=n,
        p=[0.18, 0.18, 0.18, 0.12, 0.12, 0.12, 0.10],
    )

    latent_score = (
        0.14 * priors_count
        + 0.45 * (age < 25).astype(float)
        + 0.25 * (c_charge_degree == "F").astype(float)
        + 0.20 * juv_fel_count
        + 0.10 * juv_misd_count
        + 0.12 * (race == "African-American").astype(float)
        + rng.normal(0, 1.0, size=n)
    )

    y = (latent_score > np.quantile(latent_score, 0.55)).astype(int)

    X = pd.DataFrame(
        {
            "age": age,
            "age_cat": age_cat,
            "sex": sex,
            "race": race,
            "juv_fel_count": juv_fel_count,
            "juv_misd_count": juv_misd_count,
            "juv_other_count": juv_other_count,
            "priors_count": priors_count,
            "c_charge_degree": c_charge_degree,
            "c_charge_desc": c_charge_desc,
        }
    )

    y = pd.Series(y, name="two_year_recid")
    sensitive = pd.Series(race, name="race")

    return X, y, sensitive


def _read_compas_dataframe(
    *,
    data_path: str | None,
    download_url: str,
) -> tuple[pd.DataFrame, str]:
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"COMPAS data_path does not exist: {path}")

        return pd.read_csv(path), str(path)

    try:
        df = pd.read_csv(download_url)
        return df, download_url
    except Exception as exc:
        raise RuntimeError(
            "Could not load COMPAS from the default URL. "
            "Download compas-scores-two-years.csv manually and pass "
            "--data_path data/compas/compas-scores-two-years.csv"
        ) from exc


def _apply_compas_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    if "days_b_screening_arrest" in filtered.columns:
        filtered = filtered[
            filtered["days_b_screening_arrest"].between(-30, 30, inclusive="both")
        ]

    if "is_recid" in filtered.columns:
        filtered = filtered[filtered["is_recid"] != -1]

    if "c_charge_degree" in filtered.columns:
        filtered = filtered[filtered["c_charge_degree"] != "O"]

    if "score_text" in filtered.columns:
        filtered = filtered[filtered["score_text"] != "N/A"]

    return filtered.copy()


def load_compas(
    *,
    sample_size: int,
    random_state: int,
    data_path: str | None,
    download_url: str,
    sensitive_col: str,
    target_col: str,
    synthetic_smoke: bool,
    apply_propublica_filters: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    if synthetic_smoke:
        X, y, sensitive = make_synthetic_compas_smoke(
            n=sample_size,
            random_state=random_state,
        )

        metadata = {
            "dataset": "COMPAS synthetic smoke",
            "sample_size": int(len(X)),
            "sensitive_col": sensitive_col,
            "target_col": target_col,
            "source": "synthetic",
            "apply_propublica_filters": False,
        }

        return X, y, sensitive, metadata

    df, source = _read_compas_dataframe(
        data_path=data_path,
        download_url=download_url,
    )

    original_rows = len(df)

    if apply_propublica_filters:
        df = _apply_compas_filters(df)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    if sensitive_col not in df.columns:
        raise ValueError(
            f"Sensitive column {sensitive_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in DEFAULT_COMPAS_FEATURES if c in df.columns]

    if sensitive_col not in feature_cols:
        feature_cols.append(sensitive_col)

    feature_cols = [c for c in feature_cols if c != target_col]

    required_cols = feature_cols + [target_col]
    df = df[required_cols].copy()
    df = df.dropna(subset=[target_col, sensitive_col])

    y = df[target_col].astype(int).reset_index(drop=True)
    X = df[feature_cols].reset_index(drop=True)

    if sample_size is not None and sample_size > 0 and sample_size < len(X):
        sampled = X.copy()
        sampled["__target__"] = y.to_numpy()
        sampled = sampled.sample(n=sample_size, random_state=random_state)

        y = sampled["__target__"].astype(int).reset_index(drop=True)
        X = sampled.drop(columns=["__target__"]).reset_index(drop=True)

    sensitive = X[sensitive_col].copy()
    sensitive.name = sensitive_col

    metadata = {
        "dataset": "COMPAS",
        "sample_size": int(len(X)),
        "source": source,
        "original_rows": int(original_rows),
        "rows_after_filters": int(len(df)),
        "apply_propublica_filters": bool(apply_propublica_filters),
        "sensitive_col": sensitive_col,
        "target_col": target_col,
        "feature_columns": list(X.columns),
        "target_name": y.name,
    }

    return X, y, sensitive, metadata


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    X, y, sensitive, dataset_metadata = load_compas(
        sample_size=args.sample_size,
        random_state=args.random_state,
        data_path=args.data_path,
        download_url=args.download_url,
        sensitive_col=args.sensitive_col,
        target_col=args.target_col,
        synthetic_smoke=args.synthetic_smoke,
        apply_propublica_filters=not args.no_propublica_filters,
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

        selected_features = [c for c in selected_features if c != args.sensitive_col]

        if not selected_features:
            raise RuntimeError(
                f"{selector_name} selected no usable non-sensitive features."
            )

        X_train_selected = _transform_selected(fitted_selector, X_train, selected_features)
        X_test_selected = _transform_selected(fitted_selector, X_test, selected_features)

        X_train_selected = X_train_selected[
            [c for c in X_train_selected.columns if c != args.sensitive_col]
        ]

        X_test_selected = X_test_selected[
            [c for c in X_test_selected.columns if c != args.sensitive_col]
        ]

        selected_features = list(X_train_selected.columns)

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
                    "dataset": "COMPAS",
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

    results_path = output_dir / "compas_pilot_results.csv"
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

    metadata_path = output_dir / "compas_pilot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved results to: {results_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Rows: {len(results)}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run COMPAS pilot experiment.")

    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "gradient_boosting"],
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--output_dir", type=str, default="results/compas/final")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.3)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--download_url", type=str, default=DEFAULT_COMPAS_URL)
    parser.add_argument("--sensitive_col", type=str, default="race")
    parser.add_argument("--target_col", type=str, default="two_year_recid")

    parser.add_argument(
        "--no_propublica_filters",
        action="store_true",
        help="Disable the common ProPublica COMPAS filtering rules.",
    )

    parser.add_argument(
        "--synthetic_smoke",
        action="store_true",
        help="Use synthetic COMPAS-like data instead of the real COMPAS CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
