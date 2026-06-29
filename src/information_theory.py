"""Information-theoretic utility interfaces and placeholders."""

from __future__ import annotations

import pandas as pd


class InformationTheoryError(NotImplementedError):
    """Raised for unimplemented information-theory operations."""


def mutual_information(x: pd.Series, y: pd.Series) -> float:
    """Estimate mutual information between two variables (placeholder)."""
    raise InformationTheoryError("mutual_information is not implemented yet.")


def conditional_mutual_information(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Estimate conditional mutual information I(X;Y|Z) (placeholder)."""
    raise InformationTheoryError("conditional_mutual_information is not implemented yet.")


def entropy(x: pd.Series) -> float:
    """Estimate entropy H(X) (placeholder)."""
    raise InformationTheoryError("entropy is not implemented yet.")
