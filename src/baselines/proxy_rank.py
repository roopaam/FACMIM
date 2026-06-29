"""Proxy-leakage ranking baseline placeholder."""

from __future__ import annotations

import pandas as pd


def rank_by_proxy_leakage(X: pd.DataFrame, sensitive: pd.Series) -> pd.Series:
    """Return per-feature leakage proxy scores (placeholder)."""
    raise NotImplementedError("rank_by_proxy_leakage is not implemented yet.")


class ProxyRankBaseline:
    """Class wrapper for proxy-ranking baseline."""

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "ProxyRankBaseline":
        raise NotImplementedError("ProxyRankBaseline.fit is not implemented yet.")
