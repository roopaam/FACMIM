"""Base selector API definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd


class BaseSelector(ABC):
    """Abstract API for feature selectors using pandas inputs."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "BaseSelector":
        """Fit selector on features, target, and sensitive attribute."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input features to selected subset."""

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> pd.DataFrame:
        """Fit selector and return transformed features."""
        return self.fit(X, y, sensitive).transform(X)

    @abstractmethod
    def get_support(self) -> list[str]:
        """Return selected feature names."""

    @abstractmethod
    def get_diagnostics(self) -> Mapping[str, Any]:
        """Return selector diagnostics."""
