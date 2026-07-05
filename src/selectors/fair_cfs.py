from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from src.selectors.cmim import _discretize_series_local, _mi, _to_series


class FairCFSStyleSelector:
    """
    FairCFS-style greedy baseline.

    This is a correlation/MI-based subset scoring baseline inspired by
    Correlation-based Feature Selection.

    For a candidate subset S:

        cfs_merit(S) = k * mean I(X_j ; Y)
                       / sqrt(k + k(k - 1) * mean I(X_i ; X_j))

    Fairness-aware score:

        score(S) = cfs_merit(S) - lambda_fairness * mean I(X_j ; A)

    where:
        Y = target
        A = sensitive attribute
        k = number of features in the candidate subset

    This is a baseline, not the proposed FA-CMIM method.
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

    def _subset_stats(
        self,
        Xd: pd.DataFrame,
        yd: pd.Series,
        Ad: pd.Series | None,
        subset: list[str],
    ) -> dict[str, float]:
        k_subset = len(subset)

        if k_subset == 0:
            return {
                "subset_merit": 0.0,
                "mean_feature_relevance": 0.0,
                "mean_feature_redundancy": 0.0,
                "mean_proxy_leakage": 0.0,
                "max_proxy_leakage": 0.0,
                "score": 0.0,
            }

        relevance_values = [_mi(Xd[c], yd) for c in subset]
        mean_relevance = float(np.mean(relevance_values))

        redundancy_values = []
        for a, b in itertools.combinations(subset, 2):
            redundancy_values.append(_mi(Xd[a], Xd[b]))

        mean_redundancy = float(np.mean(redundancy_values)) if redundancy_values else 0.0

        denominator = math.sqrt(
            k_subset + k_subset * (k_subset - 1) * mean_redundancy
        )

        if denominator <= 0:
            subset_merit = 0.0
        else:
            subset_merit = float((k_subset * mean_relevance) / denominator)

        if Ad is None:
            leakage_values = [0.0 for _ in subset]
        else:
            leakage_values = [_mi(Xd[c], Ad) for c in subset]

        mean_proxy_leakage = float(np.mean(leakage_values)) if leakage_values else 0.0
        max_proxy_leakage = float(np.max(leakage_values)) if leakage_values else 0.0

        score = float(subset_merit - self.fairness_penalty * mean_proxy_leakage)

        return {
            "subset_merit": subset_merit,
            "mean_feature_relevance": mean_relevance,
            "mean_feature_redundancy": mean_redundancy,
            "mean_proxy_leakage": mean_proxy_leakage,
            "max_proxy_leakage": max_proxy_leakage,
            "score": score,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive: pd.Series | None = None,
    ) -> "FairCFSStyleSelector":
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
            individual_proxy_leakage = {c: 0.0 for c in remaining}
        else:
            individual_proxy_leakage = {c: _mi(Xd[c], Ad) for c in remaining}

        n_to_select = min(self.k, len(remaining))

        for rank in range(1, n_to_select + 1):
            scored_rows = []

            for c in remaining:
                candidate_subset = selected + [c]
                stats = self._subset_stats(Xd, yd, Ad, candidate_subset)

                scored_rows.append(
                    {
                        "candidate": c,
                        "candidate_subset_size": len(candidate_subset),
                        "marginal_relevance": float(marginal_relevance[c]),
                        "candidate_proxy_leakage": float(individual_proxy_leakage[c]),
                        "subset_merit": stats["subset_merit"],
                        "mean_feature_relevance": stats["mean_feature_relevance"],
                        "mean_feature_redundancy": stats["mean_feature_redundancy"],
                        "mean_proxy_leakage": stats["mean_proxy_leakage"],
                        "max_proxy_leakage": stats["max_proxy_leakage"],
                        "fairness_penalty": float(self.fairness_penalty),
                        "score": stats["score"],
                    }
                )

            order_index = {c: i for i, c in enumerate(self.candidate_features_)}

            scored_rows = sorted(
                scored_rows,
                key=lambda r: (
                    -r["score"],
                    -r["subset_merit"],
                    -r["marginal_relevance"],
                    r["mean_feature_redundancy"],
                    r["mean_proxy_leakage"],
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
                        "candidate_subset_size": row["candidate_subset_size"],
                        "marginal_relevance": row["marginal_relevance"],
                        "candidate_proxy_leakage": row["candidate_proxy_leakage"],
                        "subset_merit": row["subset_merit"],
                        "mean_feature_relevance": row["mean_feature_relevance"],
                        "mean_feature_redundancy": row["mean_feature_redundancy"],
                        "mean_proxy_leakage": row["mean_proxy_leakage"],
                        "max_proxy_leakage": row["max_proxy_leakage"],
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
            raise RuntimeError("FairCFSStyleSelector must be fitted before transform().")

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
            raise RuntimeError("FairCFSStyleSelector must be fitted before get_support().")

        return list(self.selected_features_)

    def get_selected_features(self) -> list[str]:
        return self.get_support()

    def get_support_mask(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("FairCFSStyleSelector must be fitted before get_support_mask().")

        return np.array(
            [c in self.selected_features_ for c in self.feature_names_in_],
            dtype=bool,
        )

    def get_diagnostics(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FairCFSStyleSelector must be fitted before get_diagnostics().")

        return self.diagnostics_.copy()

    def get_feature_scores(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FairCFSStyleSelector must be fitted before get_feature_scores().")

        return self.feature_scores_.copy()


# Backward-compatible aliases.
FairCFSSelector = FairCFSStyleSelector
FairCFSBaselineSelector = FairCFSStyleSelector
FairCFS = FairCFSStyleSelector
