"""Leakage estimator interfaces and placeholders."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


class LeakageEstimatorError(NotImplementedError):
    """Raised for unimplemented leakage estimators."""


def estimate_proxy_leakage(feature: pd.Series, sensitive: pd.Series) -> float:
    """Estimate direct leakage from a feature to sensitive attribute."""
    raise LeakageEstimatorError("estimate_proxy_leakage is not implemented yet.")


def estimate_conditional_leakage(
    feature: pd.Series,
    sensitive: pd.Series,
    conditioning_set: pd.DataFrame,
) -> float:
    """Estimate conditional leakage under an existing selected subset."""
    raise LeakageEstimatorError("estimate_conditional_leakage is not implemented yet.")


def leakage_diagnostics(selected: pd.DataFrame, sensitive: pd.Series) -> Mapping[str, float]:
    """Return aggregate leakage diagnostics for selected features."""
    raise LeakageEstimatorError("leakage_diagnostics is not implemented yet.")
