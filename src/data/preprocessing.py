"""Data preprocessing interfaces and placeholders."""

from __future__ import annotations

import pandas as pd


def basic_cleaning(X: pd.DataFrame) -> pd.DataFrame:
    """Apply basic preprocessing steps to feature matrix (placeholder)."""
    raise NotImplementedError("basic_cleaning is not implemented yet.")


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables (placeholder)."""
    raise NotImplementedError("encode_categoricals is not implemented yet.")
