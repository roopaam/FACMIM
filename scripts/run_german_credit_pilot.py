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


DEFAULT_GERMAN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)


GERMAN_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]


def make_synthetic_german_credit_smoke(
    n: int = 500,
    random_state: int = 42,
    age_threshold: int = 25,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 75, size=n)
    age_group = np.where(age < age_threshold, "young", "older")

    checking_status = rng.choice(
        ["A11", "A12", "A13", "A14"],
        size=n,
        p=[0.25, 0.30, 0.15, 0.30],
    )

    duration = np.clip(
        rng.normal(
            loc=np.where(age_group == "young", 24, 18),
            scale=8,
            size=n,
        ).round(),
        4,
        72,
    ).astype(int)

    credit_history = rng.choice(
        ["A30", "A31", "A32", "A33", "A34"],
        size=n,
        p=[0.08, 0.10, 0.52, 0.12, 0.18],
    )

    purpose = rng.choice(
        ["car", "furniture", "radio_tv", "education", "business", "repairs", "other"],
        size=n,
        p=[0.28, 0.18, 0.22, 0.08, 0.10, 0.06, 0.08],
    )

    credit_amount = np.clip(
        rng.lognormal(
            mean=np.where(age_group == "young", 7.7, 7.9),
            sigma=0.7,
            size=n,
        ),
        250,
        20000,
    ).round().astype(int)

    savings_status = rng.choice(
        ["A61", "A62", "A63", "A64", "A65"],
        size=n,
        p=[0.55, 0.18, 0.10, 0.07, 0.10],
    )

    employment = np.where(
        age_group == "young",
        rng.choice(["A71", "A72", "A73", "A74", "A75"], size=n, p=[0.20, 0.40, 0.25, 0.10, 0.05]),
        rng.choice(["A71", "A72", "A73", "A74", "A75"], size=n, p=[0.05, 0.18, 0.32, 0.25, 0.20]),
    )

    installment_commitment = rng.integers(1, 5, size=n)

    personal_status = rng.choice(
        ["A91", "A92", "A93", "A94"],
        size=n,
        p=[0.12, 0.30, 0.45, 0.13],
    )

    other_parties = rng.choice(
        ["A101", "A102", "A103"],
        size=n,
        p=[0.85, 0.05, 0.10],
    )

    residence_since = rng.integers(1, 5, size=n)

    property_magnitude = rng.choice(
        ["A121", "A122", "A123", "A124"],
        size=n,
        p=[0.28, 0.25, 0.32, 0.15],
    )

    other_payment_plans = rng.choice(
        ["A141", "A142", "A143"],
        size=n,
        p=[0.14, 0.06, 0.80],
    )

    housing = np.where(
        age_group == "young",
        rng.choice(["A151", "A152", "A153"], size=n, p=[0.35, 0.55, 0.10]),
        rng.choice(["A151", "A152", "A153"], size=n, p=[0.12, 0.78, 0.10]),
    )

    existing_credits = rng.integers(1, 4, size=n)

    job = rng.choice(
        ["A171", "A172", "A173", "A174"],
        size=n,
        p=[0.04, 0.22, 0.63, 0.11],
    )

    num_dependents = rng.choice([1, 2], size=n, p=[0.82, 0.18])

    own_telephone = rng.choice(["A191", "A192"], size=n, p=[0.62, 0.38])

    foreign_worker = rng.choice(["A201", "A202"], size=n, p=[0.96, 0.04])

    risk_score = (
        0.75 * (checking_status == "A11").astype(float)
        + 0.45 * (savings_status == "A61").astype(float)
        + 0.35 * (employment == "A71").astype(float)
        + 0.025 * duration
        + 0.00006 * credit_amount
        + 0.35 * (housing == "A151").astype(float)
        + 0.25 * (age_group == "young").astype(float)
        - 0.45 * (credit_history == "A34").astype(float)
        + rng.normal(0, 0.9, size=n)
    )

    y = (risk_score > np.quantile(risk_score, 0.70)).astype(int)

    X = pd.DataFrame(
        {
            "checking_status": checking_status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "savings_status": savings_status,
            "employment": employment,
            "installment_commitment": installment_commitment,
            "personal_status": personal_status,
            "other_parties": other_parties,
            "residence_since": residence_since,
            "property_magnitude": property_magnitude,
            "other_payment_plans": other_payment_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "num_dependents": num_dependents,
            "own_telephone": own_telephone,
            "foreign_worker": foreign_worker,
        }
    )

    y = pd.Series(y, name="bad_credit")
    sensitive = pd.Series(age_group, name="age_group")

    return X, y, sensitive


def _read_german_dataframe(
    *,
    data_path: str | None,
    download_url: str,
) -> tuple[pd.DataFrame, str]:
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"German Credit data_path does not exist: {path}")

        return pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=GERMAN_COLUMNS,
            engine="python",
        ), str(path)

    try:
        df = pd.read_csv(
            download_url,
            sep=r"\s+",
            header=None,
            names=GERMAN_COLUMNS,
            engine="python",
        )
        return df, download_url
    except Exception as exc:
        raise RuntimeError(
            "Could not load German Credit from the default UCI URL. "
            "Download german.data manually and pass "
            "--data_path data/german_credit/german.data"
        ) from exc


def load_german_credit(
    *,
    sample_size: int,
    random_state: int,
    data_path: str | None,
    download_url: str,
    age_threshold: int,
    synthetic_smoke: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    if synthetic_smoke:
        X, y, sensitive = make_synthetic_german_credit_smoke(
            n=sample_size,
            random_state=random_state,
            age_threshold=age_threshold,
        )

        metadata = {
            "dataset": "German Credit synthetic smoke",
            "sample_size": int(len(X)),
            "sensitive_col": "age_group",
            "target_col": "bad_credit",
            "source": "synthetic",
            "age_threshold": int(age_threshold),
        }

        return X, y, sensitive, metadata

    df, source = _read_german_dataframe(
        data_path=data_path,
        download_url=download_url,
    )

    original_rows = len(df)

    df = df.dropna(subset=["class", "age"]).copy()

    # UCI German Credit coding:
    # class = 1 means good credit, class = 2 means bad credit.
    y = (df["class"].astype(int) == 2).astype(int)
    y = pd.Series(y.to_numpy(), name="bad_credit")

    age_group = np.where(df["age"].astype(float) < age_threshold, "young", "older")
    sensitive = pd.Series(age_group, name="age_group")

    # Remove the raw protected attribute source age from predictors.
    # The experiment tests whether other features still reconstruct age_group.
    X = df.drop(columns=["class", "age"]).reset_index(drop=True)
    y = y.reset_index(drop=True)
    sensitive = sensitive.reset_index(drop=True)

    if sample_size is not None and sample_size > 0 and sample_size < len(X):
        sampled = X.copy()
        sampled["__target__"] = y.to_numpy()
        sampled["__sensitive__"] = sensitive.to_numpy()

        sampled = sampled.sample(n=sample_size, random_state=random_state)

        y = sampled["__target__"].astype(int).reset_index(drop=True)
        sensitive = sampled["__sensitive__"].reset_index(drop=True)
        sensitive.name = "age_group"
        X = sampled.drop(columns=["__target__", "__sensitive__"]).reset_index(drop=True)

    metadata = {
        "dataset": "German Credit",
        "sample_size": int(len(X)),
        "source": source,
        "original_rows": int(original_rows),
        "sensitive_col": "age_group",
        "target_col": "bad_credit",
        "age_threshold": int(age_threshold),
        "feature_columns": list(X.columns),
        "target_name": y.name,
    }

    return X, y, sensitive, metadata


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    X, y, sensitive, dataset_metadata = load_german_credit(
        sample_size=args.sample_size,
        random_state=args.random_state,
        data_path=args.data_path,
        download_url=args.download_url,
        age_threshold=args.age_threshold,
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

        # Defensive exclusion.
        selected_features = [
            c for c in selected_features
            if c not in {"age", "age_group"}
        ]

        if not selected_features:
            raise RuntimeError(
                f"{selector_name} selected no usable non-sensitive features."
            )

        X_train_selected = _transform_selected(fitted_selector, X_train, selected_features)
        X_test_selected = _transform_selected(fitted_selector, X_test, selected_features)

        X_train_selected = X_train_selected[
            [c for c in X_train_selected.columns if c not in {"age", "age_group"}]
        ]

        X_test_selected = X_test_selected[
            [c for c in X_test_selected.columns if c not in {"age", "age_group"}]
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
                    "dataset": "German Credit",
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

    results_path = output_dir / "german_credit_pilot_results.csv"
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

    metadata_path = output_dir / "german_credit_pilot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved results to: {results_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Rows: {len(results)}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run German Credit pilot experiment.")

    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "gradient_boosting"],
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--output_dir", type=str, default="results/german_credit/final")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.3)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--download_url", type=str, default=DEFAULT_GERMAN_URL)

    parser.add_argument(
        "--age_threshold",
        type=int,
        default=25,
        help="Age threshold used to construct binary sensitive attribute age_group.",
    )

    parser.add_argument(
        "--synthetic_smoke",
        action="store_true",
        help="Use synthetic German-Credit-like data instead of the real UCI German Credit file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
