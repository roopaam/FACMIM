import numpy as np
import pandas as pd

from src.selectors.fa_cmim_basic import (
    BasicFACMIMSelector,
    FACMIMBasicSelector,
)


def make_data(n=500, seed=42):
    rng = np.random.default_rng(seed)

    sex = rng.integers(0, 2, size=n)
    fair_signal = rng.integers(0, 2, size=n)
    direct_proxy = sex.copy()
    weak_proxy = np.where(rng.random(n) < 0.8, sex, 1 - sex)
    noise = rng.integers(0, 2, size=n)

    y = ((fair_signal + direct_proxy) >= 1).astype(int)

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


def test_basic_facmim_fit_does_not_raise_notimplemented():
    X, y, sensitive = make_data()

    selector = FACMIMBasicSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_


def test_basic_facmim_alias_works():
    X, y, sensitive = make_data()

    selector = BasicFACMIMSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_
    assert len(selector.get_selected_features()) == 3


def test_basic_facmim_selects_features():
    X, y, sensitive = make_data()

    selector = FACMIMBasicSelector(k=3, fairness_penalty=0.5)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 3
    assert all(c in X.columns for c in selected)


def test_basic_facmim_excludes_sensitive_column():
    X, y, sensitive = make_data()

    selector = FACMIMBasicSelector(
        k=3,
        fairness_penalty=1.0,
        exclude_sensitive_from_candidates=True,
    )
    selector.fit(X, y, sensitive=sensitive)

    assert "sex" not in selector.get_selected_features()


def test_basic_facmim_high_penalty_avoids_direct_proxy_first():
    X, y, sensitive = make_data()

    selector = FACMIMBasicSelector(k=1, fairness_penalty=2.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] == "fair_signal"
    assert selected[0] != "direct_proxy"


def test_basic_facmim_diagnostics_include_fairness_terms():
    X, y, sensitive = make_data()

    selector = FACMIMBasicSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    expected_cols = {
        "candidate",
        "selected",
        "selection_rank",
        "marginal_relevance",
        "cmim_score",
        "utility_score",
        "fairness_leakage",
        "fairness_penalty",
        "score",
        "runtime_elapsed_seconds",
    }

    assert not diag.empty
    assert expected_cols.issubset(set(diag.columns))


def test_basic_facmim_transform_returns_original_columns():
    X, y, sensitive = make_data()

    selector = FACMIMBasicSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    Xt = selector.transform(X)

    assert list(Xt.columns) == selector.get_selected_features()
    assert len(Xt) == len(X)


def test_basic_facmim_support_mask_length():
    X, y, sensitive = make_data()

    selector = FACMIMBasicSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    mask = selector.get_support_mask()

    assert len(mask) == X.shape[1]
