"""Fairness-aware subset-aware CMIM selector placeholder."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .base import BaseSelector


class FACMIMSubsetAwareSelector(BaseSelector):
    """Placeholder for subset-aware fairness-constrained CMIM."""

    def __init__(self, n_features_to_select: int = 10, subset_penalty: float = 1.0) -> None:
        self.n_features_to_select = n_features_to_select
        self.subset_penalty = subset_penalty
        self._support: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "FACMIMSubsetAwareSelector":
        raise NotImplementedError("FACMIMSubsetAwareSelector.fit is not implemented yet.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("FACMIMSubsetAwareSelector.transform is not implemented yet.")

    def get_support(self) -> list[str]:
        return list(self._support)

    def get_diagnostics(self) -> Mapping[str, Any]:
        return {"selector": "fa_cmim_subset", "implemented": False}
