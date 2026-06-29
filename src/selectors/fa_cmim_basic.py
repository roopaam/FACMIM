"""Fairness-aware basic CMIM selector placeholder."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .base import BaseSelector


class FACMIMBasicSelector(BaseSelector):
    """Placeholder for fairness-aware CMIM baseline."""

    def __init__(self, n_features_to_select: int = 10, fairness_tradeoff: float = 0.5) -> None:
        self.n_features_to_select = n_features_to_select
        self.fairness_tradeoff = fairness_tradeoff
        self._support: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "FACMIMBasicSelector":
        raise NotImplementedError("FACMIMBasicSelector.fit is not implemented yet.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("FACMIMBasicSelector.transform is not implemented yet.")

    def get_support(self) -> list[str]:
        return list(self._support)

    def get_diagnostics(self) -> Mapping[str, Any]:
        return {"selector": "fa_cmim_basic", "implemented": False}
