"""Fair LASSO baseline placeholder. Model training intentionally deferred."""

from __future__ import annotations

import pandas as pd


class FairLassoBaseline:
    """Placeholder for fair LASSO baseline."""

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "FairLassoBaseline":
        raise NotImplementedError("FairLassoBaseline.fit is not implemented yet.")
