from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.selectors.cmim import _discretize_series_local, _mi, _to_series


class ProxyRankSelector:
    """
    Marginal proxy-ranking baseline.

    This baseline ranks each candidate feature using:

        score_j = I(X_j ; Y) - lambda_fairness * I(X_j ; A)

    where:
        Y = target
        A = sensitive attribute

    It is intentionally marginal and does not use conditional redundancy,
    subset-aware leakage, or CMIM-style conditional utility.
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
        self.proxy_ranking_: pd.DataFrame | None = None
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
    ) -> "ProxyRankSelector":
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

        rows: list[dict[str, Any]] = []

        for candidate in self.candidate_features_:
            relevance = _mi(Xd[candidate], yd)

            if Ad is None:
                proxy_leakage = 0.0
            else:
                proxy_leakage = _mi(Xd[candidate], Ad)

            score = float(relevance - self.fairness_penalty * proxy_leakage)

            rows.append(
                {
                    "candidate": candidate,
                    "marginal_relevance": float(relevance),
                    "proxy_leakage": float(proxy_leakage),
                    "fairness_penalty": float(self.fairness_penalty),
                    "score": score,
                }
            )

        order_index = {c: i for i, c in enumerate(self.candidate_features_)}

        ranked = sorted(
            rows,
            key=lambda r: (
                -r["score"],
                -r["marginal_relevance"],
                r["proxy_leakage"],
                order_index[r["candidate"]],
                r["candidate"],
            ),
        )

        proxy_ranked = sorted(
            rows,
            key=lambda r: (
                -r["proxy_leakage"],
                -r["marginal_relevance"],
                order_index[r["candidate"]],
                r["candidate"],
            ),
        )

        n_to_select = min(self.k, len(ranked))
        selected = [r["candidate"] for r in ranked[:n_to_select]]

        selected_set = set(selected)
        selected_rank = {feature: i + 1 for i, feature in enumerate(selected)}
        proxy_rank = {row["candidate"]: i + 1 for i, row in enumerate(proxy_ranked)}
        score_rank = {row["candidate"]: i + 1 for i, row in enumerate(ranked)}

        diagnostics_rows = []
        for row in ranked:
            candidate = row["candidate"]
            diagnostics_rows.append(
                {
                    "candidate": candidate,
                    "selected": candidate in selected_set,
                    "selection_rank": selected_rank.get(candidate, np.nan),
                    "score_rank": score_rank[candidate],
                    "proxy_rank": proxy_rank[candidate],
                    "marginal_relevance": row["marginal_relevance"],
                    "proxy_leakage": row["proxy_leakage"],
                    "fairness_penalty": row["fairness_penalty"],
                    "score": row["score"],
                    "runtime_elapsed_seconds": time.time() - start,
                }
            )

        self.selected_features_ = selected
        self.selection_order_ = selected.copy()
        self.runtime_seconds_ = time.time() - start
        self.diagnostics_ = pd.DataFrame(diagnostics_rows)
        self.feature_scores_ = self.diagnostics_.copy()

        self.proxy_ranking_ = (
            self.diagnostics_
            .sort_values(["proxy_leakage", "marginal_relevance"], ascending=[False, False])
            .reset_index(drop=True)
        )

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("ProxyRankSelector must be fitted before transform().")

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
            raise RuntimeError("ProxyRankSelector must be fitted before get_support().")

        return list(self.selected_features_)

    def get_selected_features(self) -> list[str]:
        return self.get_support()

    def get_support_mask(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("ProxyRankSelector must be fitted before get_support_mask().")

        return np.array(
            [c in self.selected_features_ for c in self.feature_names_in_],
            dtype=bool,
        )

    def get_diagnostics(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("ProxyRankSelector must be fitted before get_diagnostics().")

        return self.diagnostics_.copy()

    def get_feature_scores(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("ProxyRankSelector must be fitted before get_feature_scores().")

        return self.feature_scores_.copy()

    def get_proxy_ranking(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("ProxyRankSelector must be fitted before get_proxy_ranking().")

        return self.proxy_ranking_.copy()


# Backward-compatible alias.
MarginalProxyRankSelector = ProxyRankSelector
