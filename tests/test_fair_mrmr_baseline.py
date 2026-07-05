import numpy as np
import pandas as pd

from src.selectors.fair_mrmr import (
    FairMRMR,
    FairMRMRBaselineSelector,
    FairMRMRSelector,
    FairmRMRSelector,
)


def make_redundancy_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    useful = rng.integers(0, 2, size=n)
    independent_useful = rng.integers(0, 2, size=n)
    redundant = useful.copy()
    noise = rng.integers(0, 2, size=n)
    sex = rng.integers(0, 2, size=n)

    y = ((useful + independent_useful) >= 1).astype(int)

    X = pd.DataFrame(
        {
            "useful": useful,
            "redundant": redundant,
            "independent_useful": independent_useful,
            "noise": noise,
            "sex": sex,
        }
    )

    return X, pd.Series(y, name="target"), pd.Series(sex, name="sex")


def make_fairness_data(n=600, seed=42):
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


def test_fair_mrmr_fit_does_not_raise_notimplemented():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_


def test_fair_mrmr_aliases_work():
    X, y, sensitive = make_fairness_data()

    aliases = [
        FairmRMRSelector,
        FairMRMRBaselineSelector,
        FairMRMR,
    ]

    for cls in aliases:
        selector = cls(k=2, fairness_penalty=1.0)
        selector.fit(X, y, sensitive=sensitive)

        assert selector.is_fitted_
        assert len(selector.get_selected_features()) == 2


def test_fair_mrmr_selects_requested_number_of_features():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(k=3, fairness_penalty=0.5)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 3
    assert all(c in X.columns for c in selected)


def test_fair_mrmr_excludes_sensitive_column():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(
        k=3,
        fairness_penalty=1.0,
        exclude_sensitive_from_candidates=True,
    )
    selector.fit(X, y, sensitive=sensitive)

    assert "sex" not in selector.get_selected_features()


def test_fair_mrmr_high_penalty_avoids_direct_proxy_first():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(k=1, fairness_penalty=2.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] == "fair_signal"
    assert selected[0] != "direct_proxy"


def test_fair_mrmr_with_zero_fairness_penalty_behaves_like_mrmr_redundancy():
    X, y, sensitive = make_redundancy_data()

    selector = FairMRMRSelector(
        k=2,
        fairness_penalty=0.0,
        redundancy_weight=1.0,
    )
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] == "useful"
    assert selected[1] == "independent_useful"
    assert "redundant" not in selected[:2]


def test_fair_mrmr_diagnostics_include_fairness_and_redundancy_terms():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    expected_cols = {
        "candidate",
        "selected",
        "selection_rank",
        "marginal_relevance",
        "redundancy",
        "redundancy_weight",
        "proxy_leakage",
        "fairness_penalty",
        "score",
        "runtime_elapsed_seconds",
    }

    assert not diag.empty
    assert expected_cols.issubset(set(diag.columns))


def test_fair_mrmr_transform_returns_original_columns():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    Xt = selector.transform(X)

    assert list(Xt.columns) == selector.get_selected_features()
    assert len(Xt) == len(X)


def test_fair_mrmr_support_mask_length():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    mask = selector.get_support_mask()

    assert len(mask) == X.shape[1]


def test_fair_mrmr_without_sensitive_sets_zero_proxy_leakage():
    X, y, _ = make_fairness_data()

    selector = FairMRMRSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=None)

    diag = selector.get_diagnostics()

    assert diag["proxy_leakage"].max() == 0.0


def test_fair_mrmr_k_larger_than_candidates_is_safe():
    X, y, sensitive = make_fairness_data()

    selector = FairMRMRSelector(k=100, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 4
    assert "sex" not in selected
