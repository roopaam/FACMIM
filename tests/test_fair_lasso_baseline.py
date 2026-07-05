import numpy as np
import pandas as pd

from src.selectors.fair_lasso import (
    FairLasso,
    FairLassoBaselineSelector,
    FairLassoSelector,
    FairLassoStyleSelector,
)


def make_fairness_data(n=700, seed=42):
    rng = np.random.default_rng(seed)

    sex = rng.integers(0, 2, size=n)
    fair_signal = rng.integers(0, 2, size=n)
    direct_proxy = sex.copy()
    weak_proxy = np.where(rng.random(n) < 0.8, sex, 1 - sex)
    noise = rng.integers(0, 2, size=n)

    y = ((fair_signal + direct_proxy + rng.binomial(1, 0.05, size=n)) >= 1).astype(int)

    X = pd.DataFrame(
        {
            "direct_proxy": direct_proxy,
            "fair_signal": fair_signal,
            "weak_proxy": weak_proxy,
            "noise": noise,
            "sex": sex,
        }
    )

    return X, pd.Series(y, name="target"), pd.Series(sex, name="sex")


def test_fair_lasso_fit_does_not_raise_notimplemented():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_


def test_fair_lasso_aliases_work():
    X, y, sensitive = make_fairness_data()

    aliases = [
        FairLassoSelector,
        FairLassoBaselineSelector,
        FairLasso,
    ]

    for cls in aliases:
        selector = cls(k=2, fairness_penalty=1.0)
        selector.fit(X, y, sensitive=sensitive)

        assert selector.is_fitted_
        assert len(selector.get_selected_features()) == 2


def test_fair_lasso_selects_requested_number_of_features():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=3, fairness_penalty=0.5)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 3
    assert all(c in X.columns for c in selected)


def test_fair_lasso_excludes_sensitive_column():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(
        k=3,
        fairness_penalty=1.0,
        exclude_sensitive_from_candidates=True,
    )
    selector.fit(X, y, sensitive=sensitive)

    assert "sex" not in selector.get_selected_features()


def test_fair_lasso_high_penalty_avoids_direct_proxy_first():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=1, fairness_penalty=2.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] == "fair_signal"
    assert selected[0] != "direct_proxy"


def test_fair_lasso_zero_penalty_prefers_predictive_feature():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=1, fairness_penalty=0.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] in {"direct_proxy", "fair_signal"}


def test_fair_lasso_diagnostics_include_lasso_and_fairness_terms():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    expected_cols = {
        "candidate",
        "selected",
        "selection_rank",
        "score_rank",
        "lasso_importance",
        "marginal_relevance",
        "proxy_leakage",
        "fairness_penalty",
        "nonzero_encoded_count",
        "score",
        "runtime_elapsed_seconds",
    }

    assert not diag.empty
    assert expected_cols.issubset(set(diag.columns))


def test_fair_lasso_transform_returns_original_columns():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    Xt = selector.transform(X)

    assert list(Xt.columns) == selector.get_selected_features()
    assert len(Xt) == len(X)


def test_fair_lasso_support_mask_length():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    mask = selector.get_support_mask()

    assert len(mask) == X.shape[1]


def test_fair_lasso_without_sensitive_sets_zero_proxy_leakage():
    X, y, _ = make_fairness_data()

    selector = FairLassoStyleSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=None)

    diag = selector.get_diagnostics()

    assert diag["proxy_leakage"].max() == 0.0


def test_fair_lasso_get_feature_scores_non_empty():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    scores = selector.get_feature_scores()

    assert not scores.empty
    assert "lasso_importance" in scores.columns
    assert "score" in scores.columns


def test_fair_lasso_k_larger_than_candidates_is_safe():
    X, y, sensitive = make_fairness_data()

    selector = FairLassoStyleSelector(k=100, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 4
    assert "sex" not in selected
