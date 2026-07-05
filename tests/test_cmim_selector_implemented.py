import numpy as np
import pandas as pd

from src.selectors.cmim import CMIMSelector


def make_data(n=200, seed=42):
    rng = np.random.default_rng(seed)

    useful = rng.integers(0, 2, size=n)
    independent_useful = rng.integers(0, 2, size=n)
    noise = rng.integers(0, 2, size=n)
    sex = rng.integers(0, 2, size=n)
    redundant = useful.copy()

    y = ((useful + independent_useful + rng.binomial(1, 0.1, size=n)) >= 1).astype(int)

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


def test_cmim_fit_does_not_raise_notimplemented():
    X, y, sensitive = make_data()

    selector = CMIMSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_


def test_cmim_selects_features():
    X, y, sensitive = make_data()

    selector = CMIMSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) >= 1
    assert all(c in X.columns for c in selected)


def test_cmim_excludes_sensitive_column():
    X, y, sensitive = make_data()

    selector = CMIMSelector(k=3, exclude_sensitive_from_candidates=True)
    selector.fit(X, y, sensitive=sensitive)

    assert "sex" not in selector.get_selected_features()


def test_cmim_diagnostics_non_empty():
    X, y, sensitive = make_data()

    selector = CMIMSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    expected_cols = {
        "candidate",
        "selected",
        "selection_rank",
        "marginal_relevance",
        "cmim_score",
        "score",
        "runtime_elapsed_seconds",
    }

    assert not diag.empty
    assert expected_cols.issubset(set(diag.columns))


def test_cmim_transform_returns_original_columns():
    X, y, sensitive = make_data()

    selector = CMIMSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    Xt = selector.transform(X)

    assert list(Xt.columns) == selector.get_selected_features()
    assert len(Xt) == len(X)


def test_cmim_support_mask_length():
    X, y, sensitive = make_data()

    selector = CMIMSelector(k=3)
    selector.fit(X, y, sensitive=sensitive)

    mask = selector.get_support_mask()

    assert len(mask) == X.shape[1]
