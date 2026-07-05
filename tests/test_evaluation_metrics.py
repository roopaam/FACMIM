import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_demographic_parity,
    compute_equal_opportunity,
    compute_equalized_odds,
    compute_feature_leakage_metrics,
    compute_leakage_metrics,
    compute_sensitive_attacker_metrics,
    compute_utility_metrics,
)


def test_utility_perfect_prediction():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]

    out = compute_utility_metrics(y_true, y_pred, y_score)

    assert out["accuracy"] == pytest.approx(1.0)
    assert out["balanced_accuracy"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)
    assert out["auroc"] == pytest.approx(1.0)


def test_utility_one_class_auroc_nan():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    y_score = [0.9, 0.8, 0.7, 0.6]

    out = compute_utility_metrics(y_true, y_pred, y_score)

    assert out["accuracy"] == pytest.approx(1.0)
    assert math.isnan(out["auroc"])


def test_demographic_parity_identical_rates():
    sensitive = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [1, 1, 0, 0, 1, 1, 0, 0]

    out = compute_demographic_parity(y_pred, sensitive)

    assert out["dpd"] == pytest.approx(0.0)
    assert out["dpr"] == pytest.approx(1.0)


def test_demographic_parity_different_rates():
    sensitive = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [1, 1, 1, 1, 1, 0, 0, 0]

    out = compute_demographic_parity(y_pred, sensitive)

    assert out["dpd"] > 0
    assert out["dpr"] < 1


def test_demographic_parity_all_zero_predictions():
    sensitive = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0]

    out = compute_demographic_parity(y_pred, sensitive)

    assert out["dpd"] == pytest.approx(0.0)
    assert out["dpr"] == pytest.approx(1.0)


def test_equalized_odds_identical_groups():
    sensitive = [0, 0, 0, 0, 1, 1, 1, 1]
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0]

    out = compute_equalized_odds(y_true, y_pred, sensitive)

    assert out["equalized_odds_difference"] == pytest.approx(0.0)
    assert out["equalized_odds_ratio"] == pytest.approx(1.0)


def test_equal_opportunity_identical_groups():
    sensitive = [0, 0, 0, 0, 1, 1, 1, 1]
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0]

    out = compute_equal_opportunity(y_true, y_pred, sensitive)

    assert out["equal_opportunity_difference"] == pytest.approx(0.0)
    assert out["equal_opportunity_ratio"] == pytest.approx(1.0)


def test_equalized_odds_zero_denominators_safe():
    sensitive = [0, 0, 1, 1]
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]

    out = compute_equalized_odds(y_true, y_pred, sensitive)

    assert "equalized_odds_difference" in out
    assert "equalized_odds_ratio" in out


def test_empty_selected_features_zero_leakage():
    X = pd.DataFrame(index=range(10))
    sensitive = pd.Series([0, 1] * 5)

    out = compute_feature_leakage_metrics(X, sensitive)

    assert out["mean_selected_mi_sensitive"] == pytest.approx(0.0)
    assert out["max_selected_mi_sensitive"] == pytest.approx(0.0)
    assert out["joint_subset_mi_sensitive"] == pytest.approx(0.0)
    assert out["selected_feature_count"] == pytest.approx(0.0)


def test_direct_proxy_high_leakage():
    rng = np.random.default_rng(42)
    A = rng.integers(0, 2, size=1000)

    X = pd.DataFrame(
        {
            "proxy": A.copy(),
            "noise": rng.integers(0, 2, size=1000),
        }
    )

    sensitive = pd.Series(A, name="sensitive")

    out = compute_feature_leakage_metrics(X, sensitive)

    assert out["max_selected_mi_sensitive"] > 0.5
    assert out["joint_subset_mi_sensitive"] > 0.5


def test_sensitive_attacker_high_accuracy_on_direct_proxy():
    rng = np.random.default_rng(42)
    A = rng.integers(0, 2, size=1000)

    X = pd.DataFrame(
        {
            "proxy": A.copy(),
            "noise": rng.integers(0, 2, size=1000),
        }
    )

    sensitive = pd.Series(A, name="sensitive")

    out = compute_sensitive_attacker_metrics(X, sensitive)

    assert out["sensitive_attacker_status"] == "ok"
    assert out["sensitive_attacker_balanced_accuracy"] > 0.90


def test_sensitive_attacker_skips_no_features():
    X = pd.DataFrame(index=range(10))
    sensitive = pd.Series([0, 1] * 5)

    out = compute_sensitive_attacker_metrics(X, sensitive)

    assert out["sensitive_attacker_status"] == "skipped_no_features"


def test_sensitive_attacker_skips_one_class_sensitive():
    X = pd.DataFrame({"x": range(10)})
    sensitive = pd.Series([1] * 10)

    out = compute_sensitive_attacker_metrics(X, sensitive)

    assert out["sensitive_attacker_status"] == "skipped_single_sensitive_class"


def test_compute_leakage_metrics_combines_feature_and_attacker_metrics():
    rng = np.random.default_rng(42)
    A = rng.integers(0, 2, size=1000)
    X = pd.DataFrame({"proxy": A.copy()})
    sensitive = pd.Series(A, name="sensitive")

    out = compute_leakage_metrics(X, sensitive, include_attacker=True)

    assert "mean_selected_mi_sensitive" in out
    assert "sensitive_attacker_status" in out


def test_compute_all_metrics_with_selected_features():
    rng = np.random.default_rng(42)

    y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1, 1, 1, 0, 0])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.6, 0.7, 0.3, 0.4])
    sensitive = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    X = pd.DataFrame(
        {
            "x1": rng.integers(0, 2, size=8),
            "x2": sensitive,
        }
    )

    out = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        sensitive=sensitive,
        X_selected=X,
        include_attacker=True,
    )

    assert "accuracy" in out
    assert "dpd" in out
    assert "mean_selected_mi_sensitive" in out
    assert "sensitive_attacker_status" in out


def test_compute_all_metrics_without_selected_features():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    sensitive = [0, 0, 1, 1]

    out = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
    )

    assert "accuracy" in out
    assert "dpd" in out
    assert "mean_selected_mi_sensitive" not in out
