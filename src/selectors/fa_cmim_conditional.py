"""Fairness-aware conditional CMIM selector placeholder."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .base import BaseSelector


class FACMIMConditionalSelector(BaseSelector):
    """Placeholder for conditional fairness-aware CMIM."""

    def __init__(self, n_features_to_select: int = 10, max_allowed_leakage: float = 0.05) -> None:
        self.n_features_to_select = n_features_to_select
        self.max_allowed_leakage = max_allowed_leakage
        self._support: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "FACMIMConditionalSelector":
        raise NotImplementedError("FACMIMConditionalSelector.fit is not implemented yet.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("FACMIMConditionalSelector.transform is not implemented yet.")

    def get_support(self) -> list[str]:
        return list(self._support)

    def get_diagnostics(self) -> Mapping[str, Any]:
        return {"selector": "fa_cmim_conditional", "implemented": False}
