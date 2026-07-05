from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluation.metrics import compute_all_metrics
from src.selectors.cmim import CMIMSelector
from src.selectors.fair_mrmr import FairMRMRSelector
from src.selectors.mrmr import MRMRSelector
from src.selectors.fa_cmim_basic import FACMIMBasicSelector
from src.selectors.fa_cmim_subset import FACMIMSubsetAwareSelector
from src.selectors.proxy_rank import ProxyRankSelector


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_adult_from_uci(sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    train = pd.read_csv(
        train_url,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )

    test = pd.read_csv(
        test_url,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|",
    )

    df = pd.concat([train, test], ignore_index=True)
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()

    df = df.dropna().reset_index(drop=True)

    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return df


def make_synthetic_adult_like(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sex_binary = rng.integers(0, 2, size=n)
    sex = np.where(sex_binary == 1, "Male", "Female")

    fair_signal = rng.integers(0, 2, size=n)
    proxy_relationship = np.where(
        sex_binary == 1,
        rng.choice(["Husband", "Not-in-family"], size=n, p=[0.75, 0.25]),
        rng.choice(["Wife", "Not-in-family"], size=n, p=[0.55, 0.45]),
    )

    age = rng.integers(18, 70, size=n)
    hours = rng.integers(20, 60, size=n)
    education_num = rng.integers(6, 16, size=n)

    y_prob = (
        0.15
        + 0.25 * fair_signal
        + 0.20 * sex_binary
        + 0.15 * (education_num >= 12)
        + 0.10 * (hours >= 40)
    )

    income_binary = rng.random(n) < np.clip(y_prob, 0.02, 0.95)
    income = np.where(income_binary, ">50K", "<=50K")

    df = pd.DataFrame(
        {
            "age": age,
            "workclass": rng.choice(["Private", "Self-emp", "Gov"], size=n),
            "fnlwgt": rng.integers(10000, 500000, size=n),
            "education": rng.choice(["HS-grad", "Bachelors", "Masters", "Some-college"], size=n),
            "education-num": education_num,
            "marital-status": np.where(sex_binary == 1, "Married-civ-spouse", "Never-married"),
            "occupation": rng.choice(["Exec-managerial", "Sales", "Tech-support", "Other-service"], size=n),
            "relationship": proxy_relationship,
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander", "Other"], size=n),
            "sex": sex,
            "capital-gain": rng.choice([0, 0, 0, 1000, 5000], size=n),
            "capital-loss": rng.choice([0, 0, 0, 500], size=n),
            "hours-per-week": hours,
            "native-country": rng.choice(["United-States", "India", "Mexico", "Philippines"], size=n),
            "income": income,
        }
    )

    return df


def make_adult_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()

    y = df["income"].astype(str).str.strip().eq(">50K").astype(int)
    y.name = "income_gt_50k"

    sensitive = df["sex"].astype(str).str.strip()
    sensitive.name = "sex"

    X = df.drop(columns=["income"]).copy()

    return X, y, sensitive


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            )
        )

    if categorical_features:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_pipe,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_model(model_name: str, random_state: int = 42) -> Pipeline:
    if model_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", "passthrough"),
            ("classifier", clf),
        ]
    )


def make_selectors(k: int, lambdas: list[float], random_state: int = 42) -> list[tuple[str, Any]]:
    selectors: list[tuple[str, Any]] = []

    selectors.append(
        (
            f"CMIM_k{k}",
            CMIMSelector(k=k, random_state=random_state),
        )
    )

    selectors.append(
        (
            f"mRMR_k{k}",
            MRMRSelector(k=k, random_state=random_state),
        )
    )

    for lam in lambdas:
        selectors.append(
            (
                f"ProxyRank_k{k}_lambda{lam}",
                ProxyRankSelector(k=k, fairness_penalty=lam, random_state=random_state),
            )
        )

        selectors.append(
            (
                f"FairmRMR_k{k}_lambda{lam}",
                FairMRMRSelector(k=k, fairness_penalty=lam, random_state=random_state),
            )
        )

        selectors.append(
            (
                f"BasicFACMIM_k{k}_lambda{lam}",
                FACMIMBasicSelector(k=k, fairness_penalty=lam, random_state=random_state),
            )
        )

        selectors.append(
            (
                f"SubsetFACMIM_k{k}_lambda{lam}",
                FACMIMSubsetAwareSelector(k=k, fairness_penalty=lam, random_state=random_state),
            )
        )

    return selectors


def evaluate_one_model(
    model_name: str,
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sensitive_test: pd.Series,
    random_state: int = 42,
) -> dict[str, Any]:
    preprocessor = build_preprocessor(X_train_selected)
    model = build_model(model_name, random_state=random_state)

    model.steps[0] = ("preprocessor", preprocessor)

    model.fit(X_train_selected, y_train)

    y_pred = pd.Series(model.predict(X_test_selected), index=X_test_selected.index)

    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test_selected)[:, 1]
        except Exception:
            y_score = y_pred
    else:
        y_score = y_pred

    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        sensitive=sensitive_test,
        X_selected=X_test_selected,
        include_attacker=True,
        random_state=random_state,
    )

    return metrics


def run_adult_pilot(
    *,
    sample_size: int = 5000,
    k: int = 8,
    lambdas: list[float] | None = None,
    model_names: list[str] | None = None,
    output_dir: str | Path = "results/adult",
    random_state: int = 42,
    synthetic_smoke: bool = False,
) -> pd.DataFrame:
    if lambdas is None:
        lambdas = [0.5, 1.0, 2.0]

    if model_names is None:
        model_names = ["logistic_regression", "random_forest", "gradient_boosting"]

    output_dir = Path(output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if synthetic_smoke:
        print("Using synthetic Adult-like smoke dataset.")
        df = make_synthetic_adult_like(n=sample_size, seed=random_state)
    else:
        print("Loading Adult Income dataset from UCI.")
        df = load_adult_from_uci(sample_size=sample_size, random_state=random_state)

    X, y, sensitive = make_adult_xy(df)

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    selectors = make_selectors(k=k, lambdas=lambdas, random_state=random_state)

    rows: list[dict[str, Any]] = []

    for selector_name, selector in selectors:
        print(f"\nFitting selector: {selector_name}")
        selector.fit(X_train, y_train, sensitive=sensitive_train)

        selected_features = selector.get_selected_features()

        print(f"Selected features: {selected_features}")

        diag = selector.get_diagnostics()
        diag_path = diagnostics_dir / f"{selector_name}_diagnostics.csv"
        diag.to_csv(diag_path, index=False)

        X_train_selected = X_train[selected_features].copy()
        X_test_selected = X_test[selected_features].copy()

        for model_name in model_names:
            print(f"  Training model: {model_name}")

            metrics = evaluate_one_model(
                model_name=model_name,
                X_train_selected=X_train_selected,
                X_test_selected=X_test_selected,
                y_train=y_train,
                y_test=y_test,
                sensitive_test=sensitive_test,
                random_state=random_state,
            )

            row = {
                "selector": selector_name,
                "model": model_name,
                "k": k,
                "lambda": getattr(selector, "fairness_penalty", np.nan),
                "selected_feature_count": len(selected_features),
                "selected_features": "|".join(selected_features),
                "diagnostics_path": str(diag_path),
            }

            row.update(metrics)
            rows.append(row)

    results = pd.DataFrame(rows)

    output_path = output_dir / "adult_pilot_results.csv"
    results.to_csv(output_path, index=False)

    metadata = {
        "sample_size": sample_size,
        "k": k,
        "lambdas": lambdas,
        "models": model_names,
        "random_state": random_state,
        "synthetic_smoke": synthetic_smoke,
        "rows": len(results),
        "output_path": str(output_path),
    }

    with open(output_dir / "adult_pilot_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved results to: {output_path}")
    print(f"Saved metadata to: {output_dir / 'adult_pilot_metadata.json'}")
    print(f"Saved diagnostics to: {diagnostics_dir}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Adult Income pilot for FA-CMIM selectors.")

    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "logistic_regression", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--output_dir", type=str, default="results/adult")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--synthetic_smoke", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "all" in args.models:
        model_names = ["logistic_regression", "random_forest", "gradient_boosting"]
    else:
        model_names = args.models

    results = run_adult_pilot(
        sample_size=args.sample_size,
        k=args.k,
        lambdas=args.lambdas,
        model_names=model_names,
        output_dir=args.output_dir,
        random_state=args.random_state,
        synthetic_smoke=args.synthetic_smoke,
    )

    display_cols = [
        "selector",
        "model",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auroc",
        "dpd",
        "dpr",
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "mean_selected_mi_sensitive",
        "joint_subset_mi_sensitive",
        "sensitive_attacker_balanced_accuracy",
        "selected_features",
    ]

    available_cols = [c for c in display_cols if c in results.columns]

    print("\nSummary:")
    print(results[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
