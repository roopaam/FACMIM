"""Data loader interfaces. Dataset downloads are intentionally not implemented."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv_dataset(path: Path, target_column: str, sensitive_column: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load a local CSV and split into X, y, sensitive."""
    raise NotImplementedError("load_csv_dataset is not implemented yet.")
