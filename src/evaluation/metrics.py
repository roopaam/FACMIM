from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


try:
    from src.information_theory import mutual_information as _base_mi
except Exception:
    _base_mi = None


def _as_series(x: Any, name: str | None = None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)

    if name is not None:
        s.name = name

    return s.reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)

    if denominator == 0.0 and numerator == 0.0:
        return 1.0
    if denominator == 0.0:
        return float("nan")

    return float(numerator / denominator)


def _range_difference(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.max(arr) - np.min(arr))


def _min_max_ratio(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 1.0

    mn = float(np.min(arr))
    mx = float(np.max(arr))
    return _safe_ratio(mn, mx)


def _discretize_series(s: pd.Series, n_bins: int = 5) -> pd.Series:
    """
    Discretize a series for mutual-information calculations.

    Low-cardinality numeric columns such as binary 0/1 variables are treated
    as categorical values directly. This prevents pd.qcut from collapsing
    binary variables into a single bin and incorrectly producing MI = 0.
    """
    s = pd.Series(s).copy()

    non_missing = s.dropna()
    nunique = non_missing.nunique(dropna=True)

    if nunique <= 1:
        return s.astype("object").where(s.notna(), "__missing__").astype(str)

    if pd.api.types.is_numeric_dtype(s):
        if nunique <= n_bins:
            return s.astype("object").where(s.notna(), "__missing__").astype(str)

        try:
            out = pd.qcut(s, q=n_bins, duplicates="drop")
            return out.astype("object").where(pd.notna(out), "__missing__").astype(str)
        except Exception:
            try:
                out = pd.cut(s, bins=n_bins, duplicates="drop")
                return out.astype("object").where(pd.notna(out), "__missing__").astype(str)
            except Exception:
                return s.astype("object").where(s.notna(), "__missing__").astype(str)

    return s.astype("object").where(s.notna(), "__missing__").astype(str)


def _estimate_mi(x: pd.Series, y: pd.Series) -> float:
    """
    Estimate mutual information for evaluation/leakage metrics.

    This intentionally uses sklearn.metrics.mutual_info_score directly so
    evaluation metrics do not depend on scaffold placeholder functions.
    """
    x = _discretize_series(_as_series(x))
    y = _discretize_series(_as_series(y))

    return float(mutual_info_score(x, y))


def _encode_joint_state(X_subset: pd.DataFrame) -> pd.Series:
    X_subset = pd.DataFrame(X_subset)

    if X_subset.shape[1] == 0:
        return pd.Series(["__empty__"] * len(X_subset))

    Xs = X_subset.copy()
    for col in Xs.columns:
        Xs[col] = _discretize_series(Xs[col])

    return Xs.apply(lambda row: "||".join(row.values.astype(str)), axis=1)


def _estimate_joint_mi(X_subset: pd.DataFrame, sensitive: pd.Series) -> float:
    X_subset = pd.DataFrame(X_subset)

    if X_subset.shape[1] == 0:
        return 0.0

    joint_state = _encode_joint_state(X_subset)
    return _estimate_mi(joint_state, sensitive)


def compute_utility_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
) -> dict[str, float]:
    y_true = _as_series(y_true)
    y_pred = _as_series(y_pred)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float("nan"),
    }

    if y_score is not None:
        try:
            if y_true.nunique(dropna=False) >= 2:
                out["auroc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["auroc"] = float("nan")

    return out


def compute_demographic_parity(
    y_pred: Any,
    sensitive: Any,
) -> dict[str, float]:
    y_pred = _as_series(y_pred)
    sensitive = _as_series(sensitive)

    rates = []
    for group in sorted(sensitive.unique()):
        mask = sensitive == group
        if mask.sum() == 0:
            continue
        rates.append(float(np.mean(y_pred[mask] == 1)))

    if not rates:
        return {
            "dpd": float("nan"),
            "dpr": float("nan"),
            "selection_rate_min": float("nan"),
            "selection_rate_max": float("nan"),
        }

    return {
        "dpd": _range_difference(rates),
        "dpr": _min_max_ratio(rates),
        "selection_rate_min": float(np.min(rates)),
        "selection_rate_max": float(np.max(rates)),
    }


def _group_tpr_fpr(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive: pd.Series,
) -> tuple[list[float], list[float]]:
    tprs = []
    fprs = []

    for group in sorted(sensitive.unique()):
        group_mask = sensitive == group

        pos_mask = group_mask & (y_true == 1)
        neg_mask = group_mask & (y_true == 0)

        if pos_mask.sum() > 0:
            tprs.append(float(np.mean(y_pred[pos_mask] == 1)))
        else:
            tprs.append(float("nan"))

        if neg_mask.sum() > 0:
            fprs.append(float(np.mean(y_pred[neg_mask] == 1)))
        else:
            fprs.append(float("nan"))

    return tprs, fprs


def compute_equalized_odds(
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
) -> dict[str, float]:
    y_true = _as_series(y_true)
    y_pred = _as_series(y_pred)
    sensitive = _as_series(sensitive)

    tprs, fprs = _group_tpr_fpr(y_true, y_pred, sensitive)

    tpr_diff = _range_difference(tprs)
    fpr_diff = _range_difference(fprs)

    tpr_ratio = _min_max_ratio(tprs)
    fpr_ratio = _min_max_ratio(fprs)

    ratios = [r for r in [tpr_ratio, fpr_ratio] if np.isfinite(r)]
    eo_ratio = float(np.min(ratios)) if ratios else 1.0

    finite_tprs = np.array([v for v in tprs if np.isfinite(v)], dtype=float)
    finite_fprs = np.array([v for v in fprs if np.isfinite(v)], dtype=float)

    return {
        "equalized_odds_difference": float(max(tpr_diff, fpr_diff)),
        "equalized_odds_ratio": eo_ratio,
        "tpr_min": float(np.min(finite_tprs)) if finite_tprs.size else float("nan"),
        "tpr_max": float(np.max(finite_tprs)) if finite_tprs.size else float("nan"),
        "fpr_min": float(np.min(finite_fprs)) if finite_fprs.size else float("nan"),
        "fpr_max": float(np.max(finite_fprs)) if finite_fprs.size else float("nan"),
    }


def compute_equal_opportunity(
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
) -> dict[str, float]:
    y_true = _as_series(y_true)
    y_pred = _as_series(y_pred)
    sensitive = _as_series(sensitive)

    tprs, _ = _group_tpr_fpr(y_true, y_pred, sensitive)

    return {
        "equal_opportunity_difference": _range_difference(tprs),
        "equal_opportunity_ratio": _min_max_ratio(tprs),
    }


def compute_fairness_metrics(
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
) -> dict[str, float]:
    out = {}
    out.update(compute_demographic_parity(y_pred=y_pred, sensitive=sensitive))
    out.update(compute_equalized_odds(y_true=y_true, y_pred=y_pred, sensitive=sensitive))
    out.update(compute_equal_opportunity(y_true=y_true, y_pred=y_pred, sensitive=sensitive))
    return out


def compute_feature_leakage_metrics(
    X_selected: pd.DataFrame | None,
    sensitive: Any,
) -> dict[str, float]:
    sensitive = _as_series(sensitive)

    if X_selected is None:
        X_selected = pd.DataFrame()

    X_selected = pd.DataFrame(X_selected)

    if X_selected.shape[1] == 0:
        return {
            "mean_selected_mi_sensitive": 0.0,
            "max_selected_mi_sensitive": 0.0,
            "joint_subset_mi_sensitive": 0.0,
            "selected_feature_count": 0.0,
        }

    mi_values = []
    for col in X_selected.columns:
        mi_values.append(_estimate_mi(X_selected[col], sensitive))

    return {
        "mean_selected_mi_sensitive": float(np.mean(mi_values)),
        "max_selected_mi_sensitive": float(np.max(mi_values)),
        "joint_subset_mi_sensitive": float(_estimate_joint_mi(X_selected, sensitive)),
        "selected_feature_count": float(X_selected.shape[1]),
    }


def compute_sensitive_attacker_metrics(
    X_selected: pd.DataFrame | None,
    sensitive: Any,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, float | str]:
    sensitive = _as_series(sensitive)

    if X_selected is None:
        X_selected = pd.DataFrame()

    X_selected = pd.DataFrame(X_selected)

    if X_selected.shape[1] == 0:
        return {
            "sensitive_attacker_balanced_accuracy": float("nan"),
            "sensitive_attacker_auroc": float("nan"),
            "sensitive_attacker_status": "skipped_no_features",
        }

    if sensitive.nunique(dropna=False) < 2:
        return {
            "sensitive_attacker_balanced_accuracy": float("nan"),
            "sensitive_attacker_auroc": float("nan"),
            "sensitive_attacker_status": "skipped_single_sensitive_class",
        }

    X = pd.get_dummies(X_selected, dummy_na=True).fillna(0.0)
    y = sensitive.astype("category").cat.codes.to_numpy()

    unique, counts = np.unique(y, return_counts=True)
    stratify = y if len(unique) >= 2 and counts.min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        clf = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )

        clf.fit(X_train, y_train)

        y_hat = clf.predict(X_test)
        bacc = float(balanced_accuracy_score(y_test, y_hat))

        auroc = float("nan")
        try:
            if len(np.unique(y_test)) >= 2 and hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_test)
                if proba.shape[1] == 2:
                    auroc = float(roc_auc_score(y_test, proba[:, 1]))
        except Exception:
            auroc = float("nan")

        return {
            "sensitive_attacker_balanced_accuracy": bacc,
            "sensitive_attacker_auroc": auroc,
            "sensitive_attacker_status": "ok",
        }

    except Exception as exc:
        return {
            "sensitive_attacker_balanced_accuracy": float("nan"),
            "sensitive_attacker_auroc": float("nan"),
            "sensitive_attacker_status": f"error: {type(exc).__name__}: {exc}",
        }


def compute_leakage_metrics(
    X_selected: pd.DataFrame | None,
    sensitive: Any,
    *,
    include_attacker: bool = True,
    random_state: int = 42,
) -> dict[str, float | str]:
    out = {}
    out.update(compute_feature_leakage_metrics(X_selected, sensitive))

    if include_attacker:
        out.update(
            compute_sensitive_attacker_metrics(
                X_selected,
                sensitive,
                random_state=random_state,
            )
        )

    return out


def compute_all_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    sensitive: Any | None = None,
    X_selected: pd.DataFrame | None = None,
    *,
    include_attacker: bool = True,
    random_state: int = 42,
) -> dict[str, float | str]:
    out = {}
    out.update(compute_utility_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score))

    if sensitive is not None:
        out.update(
            compute_fairness_metrics(
                y_true=y_true,
                y_pred=y_pred,
                sensitive=sensitive,
            )
        )

    if sensitive is not None and X_selected is not None:
        out.update(
            compute_leakage_metrics(
                X_selected=X_selected,
                sensitive=sensitive,
                include_attacker=include_attacker,
                random_state=random_state,
            )
        )

    return out
