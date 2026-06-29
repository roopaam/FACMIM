"""Model interfaces and placeholders. Training is intentionally deferred."""

from __future__ import annotations

import pandas as pd


def train_placeholder_model(X: pd.DataFrame, y: pd.Series) -> object:
    """Placeholder model training entry-point."""
    raise NotImplementedError("train_placeholder_model is not implemented yet.")
