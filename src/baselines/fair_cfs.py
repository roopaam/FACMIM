"""Fair CFS baseline placeholder."""

from __future__ import annotations

import pandas as pd


class FairCFSBaseline:
    """Placeholder for fair CFS baseline."""

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series) -> "FairCFSBaseline":
        raise NotImplementedError("FairCFSBaseline.fit is not implemented yet.")
