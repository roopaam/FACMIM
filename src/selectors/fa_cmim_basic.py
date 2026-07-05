from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.selectors.cmim import (
    _cmi,
    _discretize_series_local,
    _mi,
    _to_series,
)


class FACMIMBasicSelector:
    """
    Basic Fairness-Aware CMIM selector.

    Objective:

        score_j = utility_j - lambda_fairness * leakage_j

    where:

        utility_j =
            I(X_j ; Y)                       for first selection
            min_s I(X_j ; Y | X_s)           after features are selected

        leakage_j =
            I(X_j ; A)

    A is the sensitive attribute.
    """

    def __init__(
        self,
        k: int = 10,
        *,
        n_features: int | None = None,
        n_features_to_select: int | None = None,
        fairness_penalty: float = 1.0,
        lambda_fairness: float | None = None,
        alpha: float | None = None,
        n_bins: int = 5,
        discretize: bool = True,
        random_state: int = 42,
        exclude_sensitive_from_candidates: bool = True,
        **kwargs,
    ) -> None:
        if n_features is not None:
            k = n_features

        if n_features_to_select is not None:
            k = n_features_to_select

        if lambda_fairness is not None:
            fairness_penalty = lambda_fairness

        if alpha is not None:
            fairness_penalty = alpha

        if int(k) <= 0:
            raise ValueError("k must be positive.")

        if int(n_bins) < 2:
            raise ValueError("n_bins must be at least 2.")

        if float(fairness_penalty) < 0:
            raise ValueError("fairness_penalty must be non-negative.")

        self.k = int(k)
        self.fairness_penalty = float(fairness_penalty)
        self.lambda_fairness = float(fairness_penalty)
        self.alpha = float(fairness_penalty)
        self.n_bins = int(n_bins)
        self.discretize = bool(discretize)
        self.random_state = int(random_state)
        self.exclude_sensitive_from_candidates = bool(exclude_sensitive_from_candidates)

        self.selected_features_: list[str] = []
        self.selection_order_: list[str] = []
        self.feature_scores_: pd.DataFrame | None = None
        self.diagnostics_: pd.DataFrame | None = None
        self.runtime_seconds_: float | None = None
        self.n_features_in_: int | None = None
        self.feature_names_in_: list[str] | None = None
        self.candidate_features_: list[str] = []
        self.sensitive_name_: str | None = None
        self.is_fitted_: bool = False

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xd = X.copy()

        if self.discretize:
            for col in Xd.columns:
                Xd[col] = _discretize_series_local(Xd[col], self.n_bins)

        return Xd

    def _prepare_y(self, y: pd.Series) -> pd.Series:
        if self.discretize:
            return _discretize_series_local(y, self.n_bins)

        return y.astype("object").where(y.notna(), "__missing__").astype(str)

    def _prepare_sensitive(self, sensitive: pd.Series | None) -> pd.Series | None:
        if sensitive is None:
            return None

        if self.discretize:
            return _discretize_series_local(sensitive, self.n_bins)

        return sensitive.astype("object").where(sensitive.notna(), "__missing__").astype(str)

    def _candidate_columns(self, X: pd.DataFrame, sensitive: pd.Series | None) -> list[str]:
        candidates = list(X.columns)

        excluded_names = {"sex", "gender", "race", "sensitive", "A"}

        if sensitive is not None and getattr(sensitive, "name", None):
            self.sensitive_name_ = str(sensitive.name)
            excluded_names.add(str(sensitive.name))

        if self.exclude_sensitive_from_candidates:
            candidates = [c for c in candidates if c not in excluded_names]

        if not candidates:
            raise ValueError("No candidate features remain after sensitive/non-feature exclusion.")

        return candidates

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive: pd.Series | None = None,
    ) -> "FACMIMBasicSelector":
        start = time.time()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.columns.duplicated().any():
            raise ValueError("X contains duplicate column names.")

        y = _to_series(y, index=X.index, name="target")

        if sensitive is not None:
            sensitive = _to_series(
                sensitive,
                index=X.index,
                name=getattr(sensitive, "name", "sensitive"),
            )

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        self.candidate_features_ = self._candidate_columns(X, sensitive)

        Xd = self._prepare_X(X[self.candidate_features_])
        yd = self._prepare_y(y)
        Ad = self._prepare_sensitive(sensitive)

        selected: list[str] = []
        remaining = list(self.candidate_features_)
        diagnostics_rows: list[dict[str, Any]] = []

        marginal_relevance = {c: _mi(Xd[c], yd) for c in remaining}

        if Ad is None:
            fairness_leakage = {c: 0.0 for c in remaining}
        else:
            fairness_leakage = {c: _mi(Xd[c], Ad) for c in remaining}

        n_to_select = min(self.k, len(remaining))

        for rank in range(1, n_to_select + 1):
            scored_rows = []

            for c in remaining:
                if not selected:
                    utility_score = marginal_relevance[c]
                else:
                    utility_score = min(_cmi(Xd[c], yd, Xd[s]) for s in selected)

                leakage_score = fairness_leakage[c]
                score = float(utility_score - self.fairness_penalty * leakage_score)

                scored_rows.append(
                    {
                        "candidate": c,
                        "marginal_relevance": float(marginal_relevance[c]),
                        "cmim_score": float(utility_score),
                        "utility_score": float(utility_score),
                        "fairness_leakage": float(leakage_score),
                        "fairness_penalty": float(self.fairness_penalty),
                        "score": score,
                    }
                )

            order_index = {c: i for i, c in enumerate(self.candidate_features_)}

            scored_rows = sorted(
                scored_rows,
                key=lambda r: (
                    -r["score"],
                    -r["utility_score"],
                    -r["marginal_relevance"],
                    order_index[r["candidate"]],
                    r["candidate"],
                ),
            )

            chosen = scored_rows[0]["candidate"]
            selected.append(chosen)
            remaining.remove(chosen)

            elapsed = time.time() - start

            for row in scored_rows:
                diagnostics_rows.append(
                    {
                        "candidate": row["candidate"],
                        "selected": row["candidate"] == chosen,
                        "selection_rank": rank if row["candidate"] == chosen else np.nan,
                        "marginal_relevance": row["marginal_relevance"],
                        "cmim_score": row["cmim_score"],
                        "utility_score": row["utility_score"],
                        "fairness_leakage": row["fairness_leakage"],
                        "fairness_penalty": row["fairness_penalty"],
                        "score": row["score"],
                        "runtime_elapsed_seconds": elapsed,
                    }
                )

            if not remaining:
                break

        self.selected_features_ = selected
        self.selection_order_ = selected.copy()
        self.runtime_seconds_ = time.time() - start
        self.diagnostics_ = pd.DataFrame(diagnostics_rows)

        if not self.diagnostics_.empty:
            self.feature_scores_ = (
                self.diagnostics_
                .sort_values(["selected", "score"], ascending=[False, False])
                .drop_duplicates("candidate")
                .reset_index(drop=True)
            )
        else:
            self.feature_scores_ = pd.DataFrame()

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FACMIMBasicSelector must be fitted before transform().")

        missing = [c for c in self.selected_features_ if c not in X.columns]

        if missing:
            raise ValueError(f"Missing selected columns in X: {missing}")

        return X[self.selected_features_].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive: pd.Series | None = None,
    ) -> pd.DataFrame:
        return self.fit(X, y, sensitive=sensitive).transform(X)

    def get_support(self) -> list[str]:
        if not self.is_fitted_:
            raise RuntimeError("FACMIMBasicSelector must be fitted before get_support().")

        return list(self.selected_features_)

    def get_selected_features(self) -> list[str]:
        return self.get_support()

    def get_support_mask(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("FACMIMBasicSelector must be fitted before get_support_mask().")

        return np.array(
            [c in self.selected_features_ for c in self.feature_names_in_],
            dtype=bool,
        )

    def get_diagnostics(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FACMIMBasicSelector must be fitted before get_diagnostics().")

        return self.diagnostics_.copy()

    def get_feature_scores(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FACMIMBasicSelector must be fitted before get_feature_scores().")

        return self.feature_scores_.copy()


# Backward-compatible alias.
BasicFACMIMSelector = FACMIMBasicSelector
