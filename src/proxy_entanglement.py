
"""
Utilities for proxy-entanglement diagnostics, proxy-removal ablations,
and synthetic triangle-motif validation.

This module is intentionally self-contained so the scripts can run even if
the main experiment pipeline changes.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Information-theoretic utilities
# ---------------------------------------------------------------------

def _codes(values) -> np.ndarray:
    """
    Robustly convert any array-like values to integer category codes.

    Important: fetch_openml datasets such as Adult may contain pandas
    Categorical columns. We convert to object before fillna because pandas
    categorical arrays do not allow inserting a new missing-value label unless
    the category is pre-registered.
    """
    s = pd.Series(values).astype("object").where(pd.notna(values), "MISSING").astype(str)
    return pd.factorize(s, sort=True)[0].astype(int)


def discretize_series(s: pd.Series, n_bins: int = 5) -> np.ndarray:
    s = pd.Series(s)

    if pd.api.types.is_numeric_dtype(s):
        nunique = s.nunique(dropna=True)
        if nunique > n_bins:
            ranked = s.rank(method="first")
            try:
                binned = pd.qcut(
                    ranked,
                    q=min(n_bins, int(nunique)),
                    labels=False,
                    duplicates="drop",
                )
                return pd.Series(binned).fillna(-1).astype(int).to_numpy()
            except Exception:
                return _codes(s)
        return pd.Series(s).fillna(-999999).astype(str).pipe(_codes)

    return _codes(s)


def discretize_dataframe(X: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {c: discretize_series(X[c], n_bins=n_bins) for c in X.columns},
        index=X.index,
    )


def encode_joint_state(X: pd.DataFrame) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(len(X), dtype=int)
    if X.shape[1] == 1:
        return _codes(X.iloc[:, 0])
    joint = X.astype(str).agg("\u241f".join, axis=1)
    return _codes(joint)


def entropy(values) -> float:
    v = _codes(values)
    if len(v) == 0:
        return 0.0
    counts = np.bincount(v)
    probs = counts[counts > 0] / len(v)
    return float(-np.sum(probs * np.log2(probs)))


def mutual_information(x, y) -> float:
    x = _codes(x)
    y = _codes(y)
    xy = _codes(pd.Series(x).astype(str) + "|" + pd.Series(y).astype(str))
    return entropy(x) + entropy(y) - entropy(xy)


def conditional_mutual_information(x, y, z) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    z = _codes(z)

    total = 0.0
    n = len(z)
    for val in np.unique(z):
        mask = z == val
        if mask.sum() == 0:
            continue
        weight = mask.sum() / n
        total += weight * mutual_information(x[mask], y[mask])
    return float(total)


def joint_mi_with_sensitive(X_subset: pd.DataFrame, s) -> float:
    if X_subset.shape[1] == 0:
        return 0.0
    joint = encode_joint_state(X_subset)
    return mutual_information(joint, s)


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def _map_binary(values, positive_values: Sequence[str]) -> np.ndarray:
    positives = {str(v).strip().lower() for v in positive_values}
    return pd.Series(values).astype(str).str.strip().str.lower().isin(positives).astype(int).to_numpy()


def derive_german_sex(personal_status: pd.Series) -> np.ndarray:
    """
    German Credit personal_status values may appear as descriptive strings
    or A91-A95 codes. This maps male=1, female=0.
    """
    out = []
    for raw in personal_status.astype(str):
        v = raw.strip().lower()

        if v in {"a92", "a95"} or "female" in v:
            out.append(0)
        elif v in {"a91", "a93", "a94"} or "male" in v:
            out.append(1)
        else:
            out.append(np.nan)

    return pd.Series(out, index=personal_status.index)


def load_fairness_dataset(
    dataset: str,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    data_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Returns X, y, s, metadata.

    X excludes the sensitive attribute and direct sensitive source column.
    """
    dataset = dataset.lower()

    if data_path is not None:
        df = pd.read_csv(data_path)
    elif dataset == "adult":
        bunch = fetch_openml("adult", version=2, as_frame=True)
        df = bunch.frame.copy()
    elif dataset in {"german", "german_credit", "credit-g"}:
        bunch = fetch_openml("credit-g", version=1, as_frame=True)
        df = bunch.frame.copy()
        dataset = "german_credit"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if dataset == "adult":
        target_col = "class" if "class" in df.columns else df.columns[-1]
        sensitive_col = "sex"

        y = _map_binary(df[target_col], positive_values=[">50k", ">50k."])
        s = _map_binary(df[sensitive_col], positive_values=["male"])

        drop_cols = [target_col, sensitive_col]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

        metadata = {
            "dataset": "adult",
            "target_col": target_col,
            "sensitive_col": sensitive_col,
            "positive_target": ">50K",
            "positive_sensitive": "Male",
        }

    elif dataset in {"german_credit", "german", "credit-g"}:
        target_col = "class"
        sensitive_source_col = "personal_status"

        y = _map_binary(df[target_col], positive_values=["bad"])
        sex_series = derive_german_sex(df[sensitive_source_col])
        valid = sex_series.notna()

        df = df.loc[valid].copy()
        y = y[valid.to_numpy()]
        s = sex_series.loc[valid].astype(int).to_numpy()

        drop_cols = [target_col, sensitive_source_col]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

        metadata = {
            "dataset": "german_credit",
            "target_col": target_col,
            "sensitive_col": "derived_sex",
            "sensitive_source_col": sensitive_source_col,
            "positive_target": "bad_credit_risk",
            "positive_sensitive": "male",
        }

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    X = X.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(int)

    if sample_size is not None and sample_size > 0 and len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        s = s[idx]

    return X, y, s, metadata


# ---------------------------------------------------------------------
# Entangled proxy diagnostics
# ---------------------------------------------------------------------

def parse_selected_features(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, list):
        return [str(v) for v in value]

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass

    if "," in text:
        return [v.strip().strip("'\"") for v in text.split(",") if v.strip()]

    return [text.strip("'\"")] if text else []


def selection_frequencies_from_results(results_path: Optional[str]) -> pd.DataFrame:
    if results_path is None or not Path(results_path).exists():
        return pd.DataFrame(columns=["feature", "selection_count", "selection_frequency"])

    df = pd.read_csv(results_path)

    feature_col = None
    for c in df.columns:
        name = c.lower()
        if "selected" in name and "feature" in name:
            feature_col = c
            break

    if feature_col is None:
        return pd.DataFrame(columns=["feature", "selection_count", "selection_frequency"])

    counts = {}
    total_rows = len(df)

    for value in df[feature_col]:
        for f in parse_selected_features(value):
            counts[f] = counts.get(f, 0) + 1

    out = pd.DataFrame(
        [{"feature": f, "selection_count": c, "selection_frequency": c / max(total_rows, 1)}
         for f, c in counts.items()]
    )
    return out.sort_values("selection_count", ascending=False)


def compute_entangled_proxy_table(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    results_path: Optional[str] = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)
    sd = _codes(s)

    rows = []
    for col in X.columns:
        x = Xd[col].to_numpy()

        mi_ps = mutual_information(x, sd)
        mi_py = mutual_information(x, yd)
        cmi_py_s = conditional_mutual_information(x, yd, sd)

        # Statistical signature of entangled proxy:
        # sensitive-informative AND target-informative after conditioning on S.
        score = mi_ps * cmi_py_s

        rows.append({
            "feature": col,
            "mi_feature_sensitive": mi_ps,
            "mi_feature_target": mi_py,
            "cmi_feature_target_given_sensitive": cmi_py_s,
            "entanglement_score": score,
        })

    table = pd.DataFrame(rows)

    freqs = selection_frequencies_from_results(results_path)
    if len(freqs) > 0:
        table = table.merge(freqs, on="feature", how="left")
    else:
        table["selection_count"] = 0
        table["selection_frequency"] = 0.0

    table["selection_count"] = table["selection_count"].fillna(0).astype(int)
    table["selection_frequency"] = table["selection_frequency"].fillna(0.0)

    return table.sort_values(
        ["entanglement_score", "selection_frequency"],
        ascending=[False, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Feature selectors
# ---------------------------------------------------------------------

def select_cmim(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int,
    n_bins: int = 5,
) -> List[str]:
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)

    selected: List[str] = []
    remaining = list(X.columns)

    while len(selected) < min(k, len(remaining) + len(selected)):
        best_feature = None
        best_score = -np.inf

        for f in remaining:
            x = Xd[f].to_numpy()

            if len(selected) == 0:
                score = mutual_information(x, yd)
            else:
                score = min(
                    conditional_mutual_information(x, yd, Xd[g].to_numpy())
                    for g in selected
                )

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def select_subset_fa_cmim(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    k: int,
    lambda_value: float,
    n_bins: int = 5,
) -> List[str]:
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)
    sd = _codes(s)

    selected: List[str] = []
    remaining = list(X.columns)

    while len(selected) < min(k, len(remaining) + len(selected)):
        best_feature = None
        best_score = -np.inf

        for f in remaining:
            x = Xd[f].to_numpy()

            if len(selected) == 0:
                utility = mutual_information(x, yd)
            else:
                utility = min(
                    conditional_mutual_information(x, yd, Xd[g].to_numpy())
                    for g in selected
                )

            updated_subset = selected + [f]
            leakage = joint_mi_with_sensitive(Xd[updated_subset], sd)

            score = utility - lambda_value * leakage

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def make_model(model_name: str, random_state: int = 42):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, solver="lbfgs")
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", encoder, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _safe_auc(y_true, proba) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return np.nan


def fairness_metrics(y_true, y_pred, s) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    s = np.asarray(s).astype(int)

    groups = sorted(np.unique(s))
    if len(groups) != 2:
        return {
            "dpd": np.nan,
            "dpr": np.nan,
            "equal_opportunity_difference": np.nan,
            "equalized_odds_difference": np.nan,
        }

    g0, g1 = groups[0], groups[1]

    def rate(mask):
        return float(np.mean(y_pred[mask] == 1)) if np.sum(mask) > 0 else np.nan

    p0 = rate(s == g0)
    p1 = rate(s == g1)

    dpd = abs(p0 - p1)
    dpr = min(p0, p1) / max(p0, p1) if max(p0, p1) > 0 else np.nan

    def tpr(group):
        mask = (s == group) & (y_true == 1)
        return float(np.mean(y_pred[mask] == 1)) if np.sum(mask) > 0 else np.nan

    def fpr(group):
        mask = (s == group) & (y_true == 0)
        return float(np.mean(y_pred[mask] == 1)) if np.sum(mask) > 0 else np.nan

    tpr0, tpr1 = tpr(g0), tpr(g1)
    fpr0, fpr1 = fpr(g0), fpr(g1)

    eod = abs(tpr0 - tpr1)
    eodds = max(abs(tpr0 - tpr1), abs(fpr0 - fpr1))

    return {
        "dpd": dpd,
        "dpr": dpr,
        "equal_opportunity_difference": eod,
        "equalized_odds_difference": eodds,
    }


def attacker_balanced_accuracy(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    s_train: np.ndarray,
    s_test: np.ndarray,
    random_state: int = 42,
) -> float:
    if len(np.unique(s_train)) < 2 or len(np.unique(s_test)) < 2:
        return np.nan

    pipe = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])

    pipe.fit(X_train, s_train)
    pred = pipe.predict(X_test)
    return float(balanced_accuracy_score(s_test, pred))


def evaluate_subset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    s_train: np.ndarray,
    s_test: np.ndarray,
    selected_features: Sequence[str],
    model_name: str,
    random_state: int = 42,
    n_bins: int = 5,
) -> Dict[str, float]:
    selected_features = list(selected_features)

    Xtr = X_train[selected_features].copy()
    Xte = X_test[selected_features].copy()

    pipe = Pipeline([
        ("prep", make_preprocessor(Xtr)),
        ("clf", make_model(model_name, random_state=random_state)),
    ])

    pipe.fit(Xtr, y_train)
    pred = pipe.predict(Xte)

    try:
        proba = pipe.predict_proba(Xte)[:, 1]
    except Exception:
        proba = pred

    Xte_disc = discretize_dataframe(Xte, n_bins=n_bins)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auroc": _safe_auc(y_test, proba),
        "joint_subset_mi_sensitive": joint_mi_with_sensitive(Xte_disc, s_test),
        "sensitive_attacker_balanced_accuracy": attacker_balanced_accuracy(
            Xtr, Xte, s_train, s_test, random_state=random_state
        ),
    }

    metrics.update(fairness_metrics(y_test, pred, s_test))
    return metrics



# ---- all-algorithm ablation selectors ----

def select_mrmr(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int,
    n_bins: int = 5,
) -> List[str]:
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)

    selected: List[str] = []
    remaining = list(X.columns)

    while len(selected) < min(k, len(X.columns)):
        best_feature = None
        best_score = -np.inf

        for f in remaining:
            relevance = mutual_information(Xd[f].to_numpy(), yd)

            if selected:
                redundancy = np.mean([
                    mutual_information(Xd[f].to_numpy(), Xd[g].to_numpy())
                    for g in selected
                ])
            else:
                redundancy = 0.0

            score = relevance - redundancy

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def select_proxyrank(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    k: int,
    lambda_value: float,
    n_bins: int = 5,
) -> List[str]:
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)
    sd = _codes(s)

    rows = []
    for f in X.columns:
        relevance = mutual_information(Xd[f].to_numpy(), yd)
        leakage = mutual_information(Xd[f].to_numpy(), sd)
        score = relevance - lambda_value * leakage
        rows.append((f, score))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    return [f for f, _ in rows[:k]]


def select_fair_mrmr(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    k: int,
    lambda_value: float,
    n_bins: int = 5,
) -> List[str]:
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)
    sd = _codes(s)

    selected: List[str] = []
    remaining = list(X.columns)

    while len(selected) < min(k, len(X.columns)):
        best_feature = None
        best_score = -np.inf

        for f in remaining:
            relevance = mutual_information(Xd[f].to_numpy(), yd)
            leakage = mutual_information(Xd[f].to_numpy(), sd)

            if selected:
                redundancy = np.mean([
                    mutual_information(Xd[f].to_numpy(), Xd[g].to_numpy())
                    for g in selected
                ])
            else:
                redundancy = 0.0

            score = relevance - redundancy - lambda_value * leakage

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def select_basic_fa_cmim(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    k: int,
    lambda_value: float,
    n_bins: int = 5,
) -> List[str]:
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)
    sd = _codes(s)

    selected: List[str] = []
    remaining = list(X.columns)

    while len(selected) < min(k, len(X.columns)):
        best_feature = None
        best_score = -np.inf

        for f in remaining:
            x = Xd[f].to_numpy()

            if not selected:
                utility = mutual_information(x, yd)
            else:
                utility = min(
                    conditional_mutual_information(x, yd, Xd[g].to_numpy())
                    for g in selected
                )

            leakage = mutual_information(x, sd)
            score = utility - lambda_value * leakage

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def select_faircfs_style(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    k: int,
    lambda_value: float,
    n_bins: int = 5,
) -> List[str]:
    """
    Greedy FairCFS-style selector.

    It scores the updated subset using:
    mean target relevance - lambda * mean sensitive relevance,
    normalized by average pairwise redundancy.

    This is a correlation/information-theoretic FairCFS-style baseline,
    not a full causal-discovery implementation.
    """
    Xd = discretize_dataframe(X, n_bins=n_bins)
    yd = _codes(y)
    sd = _codes(s)

    selected: List[str] = []
    remaining = list(X.columns)

    def subset_merit(features: List[str]) -> float:
        if not features:
            return -np.inf

        relevance = np.mean([
            mutual_information(Xd[f].to_numpy(), yd)
            for f in features
        ])

        sensitive_assoc = np.mean([
            mutual_information(Xd[f].to_numpy(), sd)
            for f in features
        ])

        if len(features) <= 1:
            redundancy = 0.0
        else:
            vals = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    vals.append(
                        mutual_information(
                            Xd[features[i]].to_numpy(),
                            Xd[features[j]].to_numpy(),
                        )
                    )
            redundancy = float(np.mean(vals)) if vals else 0.0

        numerator = relevance - lambda_value * sensitive_assoc
        denominator = np.sqrt(1.0 + max(len(features) - 1, 0) * redundancy)
        return float(numerator / denominator) if denominator > 0 else float(numerator)

    while len(selected) < min(k, len(X.columns)):
        best_feature = None
        best_score = -np.inf

        for f in remaining:
            score = subset_merit(selected + [f])
            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def select_fairlasso_style(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    k: int,
    lambda_value: float,
    n_bins: int = 5,
    random_state: int = 42,
) -> List[str]:
    """
    FairLasso-style embedded selector.

    It fits an L1 logistic model, aggregates encoded coefficients back to
    original features, and subtracts a sensitive-leakage penalty.
    """
    Xd = discretize_dataframe(X, n_bins=n_bins)
    sd = _codes(s)

    X_enc = pd.get_dummies(X, dummy_na=True, prefix_sep="=", dtype=float)

    try:
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            random_state=random_state,
        )
        clf.fit(X_enc, y)
        coefs = np.abs(clf.coef_[0])
    except Exception:
        coefs = np.zeros(X_enc.shape[1])

    importance = {f: 0.0 for f in X.columns}

    for col, coef in zip(X_enc.columns, coefs):
        col_str = str(col)
        original = col_str.split("=", 1)[0]
        if original not in importance:
            original = col_str
        if original in importance:
            importance[original] = max(importance[original], float(coef))

    # If L1 collapses to all zero coefficients, fall back to target MI.
    if max(importance.values()) == 0.0:
        for f in X.columns:
            importance[f] = mutual_information(Xd[f].to_numpy(), _codes(y))

    rows = []
    for f in X.columns:
        leakage = mutual_information(Xd[f].to_numpy(), sd)
        score = importance[f] - lambda_value * leakage
        rows.append((f, score))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    return [f for f, _ in rows[:k]]


def select_all_ablation_methods(
    X: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    k: int,
    lambdas: Sequence[float],
    n_bins: int = 5,
    random_state: int = 42,
) -> List[Tuple[str, Optional[float], List[str]]]:
    """
    Returns a list of:
    (selector_name, lambda_value, selected_features)
    """
    selectors: List[Tuple[str, Optional[float], List[str]]] = []

    selectors.append((
        "CMIM",
        None,
        select_cmim(X, y, k=k, n_bins=n_bins),
    ))

    selectors.append((
        "mRMR",
        None,
        select_mrmr(X, y, k=k, n_bins=n_bins),
    ))

    for lam in lambdas:
        selectors.append((
            f"ProxyRank_lambda{lam}",
            lam,
            select_proxyrank(X, y, s, k=k, lambda_value=lam, n_bins=n_bins),
        ))

        selectors.append((
            f"FairmRMR_lambda{lam}",
            lam,
            select_fair_mrmr(X, y, s, k=k, lambda_value=lam, n_bins=n_bins),
        ))

        selectors.append((
            f"FairCFS_lambda{lam}",
            lam,
            select_faircfs_style(X, y, s, k=k, lambda_value=lam, n_bins=n_bins),
        ))

        selectors.append((
            f"FairLasso_lambda{lam}",
            lam,
            select_fairlasso_style(
                X,
                y,
                s,
                k=k,
                lambda_value=lam,
                n_bins=n_bins,
                random_state=random_state,
            ),
        ))

        selectors.append((
            f"BasicFACMIM_lambda{lam}",
            lam,
            select_basic_fa_cmim(X, y, s, k=k, lambda_value=lam, n_bins=n_bins),
        ))

        selectors.append((
            f"SubsetFACMIM_lambda{lam}",
            lam,
            select_subset_fa_cmim(X, y, s, k=k, lambda_value=lam, n_bins=n_bins),
        ))

    return selectors
