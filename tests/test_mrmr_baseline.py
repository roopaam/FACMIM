import numpy as np
import pandas as pd

from src.selectors.mrmr import MRMRBaselineSelector, MRMRSelector, mRMRSelector


def make_data(n=1000, seed=42):
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


def test_mrmr_fit_does_not_raise_notimplemented():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_


def test_mrmr_aliases_work():
    X, y, sensitive = make_data()

    aliases = [mRMRSelector, MRMRBaselineSelector]

    for cls in aliases:
        selector = cls(k=2)
        selector.fit(X, y, sensitive=sensitive)

        assert selector.is_fitted_
        assert len(selector.get_selected_features()) == 2


def test_mrmr_selects_requested_number_of_features():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 3
    assert all(c in X.columns for c in selected)


def test_mrmr_excludes_sensitive_column():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=3, exclude_sensitive_from_candidates=True)
    selector.fit(X, y, sensitive=sensitive)

    assert "sex" not in selector.get_selected_features()


def test_mrmr_prefers_independent_useful_over_redundant_copy():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=2, redundancy_weight=1.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] == "useful"
    assert selected[1] == "independent_useful"
    assert "redundant" not in selected[:2]


def test_mrmr_diagnostics_include_redundancy_terms():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=3, redundancy_weight=1.0)
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    expected_cols = {
        "candidate",
        "selected",
        "selection_rank",
        "marginal_relevance",
        "redundancy",
        "redundancy_weight",
        "score",
        "runtime_elapsed_seconds",
    }

    assert not diag.empty
    assert expected_cols.issubset(set(diag.columns))


def test_mrmr_transform_returns_original_columns():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    Xt = selector.transform(X)

    assert list(Xt.columns) == selector.get_selected_features()
    assert len(Xt) == len(X)


def test_mrmr_support_mask_length():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    mask = selector.get_support_mask()

    assert len(mask) == X.shape[1]


def test_mrmr_get_feature_scores_non_empty():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    scores = selector.get_feature_scores()

    assert not scores.empty
    assert "score" in scores.columns


def test_mrmr_k_larger_than_candidates_is_safe():
    X, y, sensitive = make_data()

    selector = MRMRSelector(k=100)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 4
    assert "sex" not in selected
