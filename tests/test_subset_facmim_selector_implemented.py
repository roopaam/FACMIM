import numpy as np
import pandas as pd

from src.selectors.fa_cmim_subset import (
    FACMIMSubsetAwareSelector,
    FACMIMSubsetSelector,
    SubsetAwareConstrainedFACMIMSelector,
    SubsetAwareFACMIMSelector,
)


def make_data(n=600, seed=42):
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


def test_subset_facmim_fit_does_not_raise_notimplemented():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_


def test_subset_facmim_aliases_work():
    X, y, sensitive = make_data()

    aliases = [
        FACMIMSubsetSelector,
        SubsetAwareFACMIMSelector,
        SubsetAwareConstrainedFACMIMSelector,
    ]

    for cls in aliases:
        selector = cls(k=2, fairness_penalty=1.0)
        selector.fit(X, y, sensitive=sensitive)

        assert selector.is_fitted_
        assert len(selector.get_selected_features()) == 2


def test_subset_facmim_selects_requested_number_of_features():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(k=3, fairness_penalty=0.5)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 3
    assert all(c in X.columns for c in selected)


def test_subset_facmim_excludes_sensitive_column():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(
        k=3,
        fairness_penalty=1.0,
        exclude_sensitive_from_candidates=True,
    )
    selector.fit(X, y, sensitive=sensitive)

    assert "sex" not in selector.get_selected_features()


def test_subset_facmim_high_penalty_avoids_direct_proxy_first():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(k=1, fairness_penalty=2.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] == "fair_signal"
    assert selected[0] != "direct_proxy"


def test_subset_facmim_diagnostics_include_subset_terms():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    expected_cols = {
        "candidate",
        "selected",
        "selection_rank",
        "marginal_relevance",
        "cmim_score",
        "utility_score",
        "candidate_individual_leakage",
        "subset_leakage",
        "current_subset_leakage",
        "incremental_subset_leakage",
        "fairness_penalty",
        "constraint_feasible",
        "score",
        "ranking_score",
        "runtime_elapsed_seconds",
    }

    assert not diag.empty
    assert expected_cols.issubset(set(diag.columns))


def test_subset_facmim_budget_marks_constraint_feasibility():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(
        k=2,
        fairness_penalty=1.0,
        max_subset_leakage=0.05,
    )
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    assert "constraint_feasible" in diag.columns
    assert diag["constraint_feasible"].isin([True, False]).all()


def test_subset_facmim_transform_returns_original_columns():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    Xt = selector.transform(X)

    assert list(Xt.columns) == selector.get_selected_features()
    assert len(Xt) == len(X)


def test_subset_facmim_support_mask_length():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    mask = selector.get_support_mask()

    assert len(mask) == X.shape[1]


def test_subset_facmim_final_subset_leakage_is_recorded():
    X, y, sensitive = make_data()

    selector = FACMIMSubsetAwareSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert isinstance(selector.final_subset_leakage_, float)
    assert selector.final_subset_leakage_ >= 0.0
