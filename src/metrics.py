"""Evaluation metric interfaces and placeholders."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Mapping[str, float]:
    """Return predictive metric dictionary (placeholder)."""
    raise NotImplementedError("evaluate_predictions is not implemented yet.")


def evaluate_fairness(y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series) -> Mapping[str, float]:
    """Return fairness metric dictionary (placeholder)."""
    raise NotImplementedError("evaluate_fairness is not implemented yet.")
