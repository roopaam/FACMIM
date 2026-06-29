"""Classical CMIM selector placeholder."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .base import BaseSelector


class CMIMSelector(BaseSelector):
    """Placeholder CMIM selector without fairness constraints."""

    def __init__(self, n_features_to_select: int = 10) -> None:
        self.n_features_to_select = n_features_to_select
        self._support: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "CMIMSelector":
        raise NotImplementedError("CMIMSelector.fit is not implemented yet.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("CMIMSelector.transform is not implemented yet.")

    def get_support(self) -> list[str]:
        return list(self._support)

    def get_diagnostics(self) -> Mapping[str, Any]:
        return {"selector": "cmim", "implemented": False}
