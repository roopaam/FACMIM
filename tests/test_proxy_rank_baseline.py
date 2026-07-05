import numpy as np
import pandas as pd

from src.selectors.proxy_rank import MarginalProxyRankSelector, ProxyRankSelector


def make_data(n=500, seed=42):
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


def test_proxy_rank_fit_does_not_raise_notimplemented():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_


def test_proxy_rank_alias_works():
    X, y, sensitive = make_data()

    selector = MarginalProxyRankSelector(k=2, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    assert selector.is_fitted_
    assert len(selector.get_selected_features()) == 2


def test_proxy_rank_selects_requested_number_of_features():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(k=3, fairness_penalty=0.5)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert len(selected) == 3
    assert all(c in X.columns for c in selected)


def test_proxy_rank_excludes_sensitive_column():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(
        k=3,
        fairness_penalty=1.0,
        exclude_sensitive_from_candidates=True,
    )
    selector.fit(X, y, sensitive=sensitive)

    assert "sex" not in selector.get_selected_features()


def test_proxy_rank_identifies_direct_proxy_as_highest_proxy_leakage():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(k=3, fairness_penalty=0.0)
    selector.fit(X, y, sensitive=sensitive)

    proxy_ranking = selector.get_proxy_ranking()

    assert proxy_ranking.iloc[0]["candidate"] == "direct_proxy"
    assert proxy_ranking.iloc[0]["proxy_leakage"] > 0.5


def test_proxy_rank_high_penalty_avoids_direct_proxy_first():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(k=1, fairness_penalty=2.0)
    selector.fit(X, y, sensitive=sensitive)

    selected = selector.get_selected_features()

    assert selected[0] == "fair_signal"
    assert selected[0] != "direct_proxy"


def test_proxy_rank_diagnostics_include_proxy_terms():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    diag = selector.get_diagnostics()

    expected_cols = {
        "candidate",
        "selected",
        "selection_rank",
        "score_rank",
        "proxy_rank",
        "marginal_relevance",
        "proxy_leakage",
        "fairness_penalty",
        "score",
        "runtime_elapsed_seconds",
    }

    assert not diag.empty
    assert expected_cols.issubset(set(diag.columns))


def test_proxy_rank_transform_returns_original_columns():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    Xt = selector.transform(X)

    assert list(Xt.columns) == selector.get_selected_features()
    assert len(Xt) == len(X)


def test_proxy_rank_support_mask_length():
    X, y, sensitive = make_data()

    selector = ProxyRankSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=sensitive)

    mask = selector.get_support_mask()

    assert len(mask) == X.shape[1]


def test_proxy_rank_without_sensitive_sets_zero_leakage():
    X, y, _ = make_data()

    selector = ProxyRankSelector(k=3, fairness_penalty=1.0)
    selector.fit(X, y, sensitive=None)

    diag = selector.get_diagnostics()

    assert diag["proxy_leakage"].max() == 0.0
