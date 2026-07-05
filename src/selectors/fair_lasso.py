from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.selectors.cmim import _discretize_series_local, _mi, _to_series


class FairLassoStyleSelector:
    """
    FairLasso-style sparse feature-selection baseline.

    This baseline fits an L1-regularized logistic model to estimate sparse
    target relevance, then penalizes candidate features that leak sensitive
    information.

    Objective:

        score_j = normalized_l1_importance_j
                  - lambda_fairness * I(X_j ; A)

    where:
        A = sensitive attribute

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
        C: float = 1.0,
        l1_C: float | None = None,
        max_iter: int = 2000,
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

        if l1_C is not None:
            C = l1_C

        if int(k) <= 0:
            raise ValueError("k must be positive.")

        if int(n_bins) < 2:
            raise ValueError("n_bins must be at least 2.")

        if float(fairness_penalty) < 0:
            raise ValueError("fairness_penalty must be non-negative.")

        if float(C) <= 0:
            raise ValueError("C must be positive.")

        self.k = int(k)
        self.fairness_penalty = float(fairness_penalty)
        self.lambda_fairness = float(fairness_penalty)
        self.alpha = float(fairness_penalty)
        self.C = float(C)
        self.l1_C = float(C)
        self.max_iter = int(max_iter)
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

    def _prepare_X_for_mi(self, X: pd.DataFrame) -> pd.DataFrame:
        Xd = X.copy()

        if self.discretize:
            for col in Xd.columns:
                Xd[col] = _discretize_series_local(Xd[col], self.n_bins)

        return Xd

    def _prepare_y_for_mi(self, y: pd.Series) -> pd.Series:
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

    def _encode_for_lasso(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        One-hot encode each original feature separately so encoded coefficients
        can be mapped back to original feature names.
        """
        encoded_parts = []
        feature_to_encoded: dict[str, list[str]] = {}

        for col in X.columns:
            s = X[col]

            if self.discretize:
                s = _discretize_series_local(s, self.n_bins)
            else:
                s = s.astype("object").where(s.notna(), "__missing__").astype(str)

            dummies = pd.get_dummies(
                s.astype(str),
                prefix=col,
                prefix_sep="__",
                dummy_na=False,
            )

            dummies = dummies.astype(float)
            encoded_parts.append(dummies)
            feature_to_encoded[col] = list(dummies.columns)

        if not encoded_parts:
            raise ValueError("No encoded features were created.")

        X_encoded = pd.concat(encoded_parts, axis=1)

        return X_encoded, feature_to_encoded

    def _lasso_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[dict[str, float], dict[str, int]]:
        X_encoded, feature_to_encoded = self._encode_for_lasso(X)

        y_codes = y.astype("category").cat.codes.to_numpy()

        if len(np.unique(y_codes)) < 2:
            return (
                {col: 0.0 for col in X.columns},
                {col: 0 for col in X.columns},
            )

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_encoded)

        clf = LogisticRegression(
            penalty="l1",
            C=self.C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        try:
            clf.fit(X_scaled, y_codes)
            coef = np.abs(clf.coef_)

            if coef.ndim == 2:
                encoded_importance = coef.max(axis=0)
            else:
                encoded_importance = coef

        except Exception:
            encoded_importance = np.zeros(X_encoded.shape[1], dtype=float)

        encoded_importance_series = pd.Series(
            encoded_importance,
            index=X_encoded.columns,
            dtype=float,
        )

        raw_importance: dict[str, float] = {}
        nonzero_counts: dict[str, int] = {}

        for feature, encoded_cols in feature_to_encoded.items():
            values = encoded_importance_series[encoded_cols]
            raw_importance[feature] = float(values.sum())
            nonzero_counts[feature] = int((values > 1e-12).sum())

        max_raw = max(raw_importance.values()) if raw_importance else 0.0

        if max_raw > 0:
            normalized = {
                feature: float(value / max_raw)
                for feature, value in raw_importance.items()
            }
        else:
            normalized = {feature: 0.0 for feature in raw_importance}

        return normalized, nonzero_counts

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive: pd.Series | None = None,
    ) -> "FairLassoStyleSelector":
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

        X_candidates = X[self.candidate_features_].copy()
        Xd = self._prepare_X_for_mi(X_candidates)
        yd = self._prepare_y_for_mi(y)
        Ad = self._prepare_sensitive(sensitive)

        lasso_importance, nonzero_counts = self._lasso_feature_importance(X_candidates, y)

        marginal_relevance = {c: _mi(Xd[c], yd) for c in self.candidate_features_}

        # Fallback for cases where L1 shrinks everything to zero.
        if max(lasso_importance.values()) <= 1e-12:
            max_mi = max(marginal_relevance.values()) if marginal_relevance else 0.0
            if max_mi > 0:
                lasso_importance = {
                    c: float(marginal_relevance[c] / max_mi)
                    for c in self.candidate_features_
                }

        if Ad is None:
            proxy_leakage = {c: 0.0 for c in self.candidate_features_}
        else:
            proxy_leakage = {c: _mi(Xd[c], Ad) for c in self.candidate_features_}

        rows: list[dict[str, Any]] = []

        for candidate in self.candidate_features_:
            score = float(
                lasso_importance[candidate]
                - self.fairness_penalty * proxy_leakage[candidate]
            )

            rows.append(
                {
                    "candidate": candidate,
                    "lasso_importance": float(lasso_importance[candidate]),
                    "marginal_relevance": float(marginal_relevance[candidate]),
                    "proxy_leakage": float(proxy_leakage[candidate]),
                    "fairness_penalty": float(self.fairness_penalty),
                    "nonzero_encoded_count": int(nonzero_counts.get(candidate, 0)),
                    "score": score,
                }
            )

        order_index = {c: i for i, c in enumerate(self.candidate_features_)}

        ranked = sorted(
            rows,
            key=lambda r: (
                -r["score"],
                -r["lasso_importance"],
                -r["marginal_relevance"],
                r["proxy_leakage"],
                order_index[r["candidate"]],
                r["candidate"],
            ),
        )

        n_to_select = min(self.k, len(ranked))
        selected = [r["candidate"] for r in ranked[:n_to_select]]
        selected_set = set(selected)
        selected_rank = {feature: i + 1 for i, feature in enumerate(selected)}

        diagnostics_rows = []
        elapsed = time.time() - start

        for rank, row in enumerate(ranked, start=1):
            candidate = row["candidate"]

            diagnostics_rows.append(
                {
                    "candidate": candidate,
                    "selected": candidate in selected_set,
                    "selection_rank": selected_rank.get(candidate, np.nan),
                    "score_rank": rank,
                    "lasso_importance": row["lasso_importance"],
                    "marginal_relevance": row["marginal_relevance"],
                    "proxy_leakage": row["proxy_leakage"],
                    "fairness_penalty": row["fairness_penalty"],
                    "nonzero_encoded_count": row["nonzero_encoded_count"],
                    "score": row["score"],
                    "runtime_elapsed_seconds": elapsed,
                }
            )

        self.selected_features_ = selected
        self.selection_order_ = selected.copy()
        self.runtime_seconds_ = time.time() - start
        self.diagnostics_ = pd.DataFrame(diagnostics_rows)
        self.feature_scores_ = self.diagnostics_.copy()
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FairLassoStyleSelector must be fitted before transform().")

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
            raise RuntimeError("FairLassoStyleSelector must be fitted before get_support().")

        return list(self.selected_features_)

    def get_selected_features(self) -> list[str]:
        return self.get_support()

    def get_support_mask(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("FairLassoStyleSelector must be fitted before get_support_mask().")

        return np.array(
            [c in self.selected_features_ for c in self.feature_names_in_],
            dtype=bool,
        )

    def get_diagnostics(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FairLassoStyleSelector must be fitted before get_diagnostics().")

        return self.diagnostics_.copy()

    def get_feature_scores(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FairLassoStyleSelector must be fitted before get_feature_scores().")

        return self.feature_scores_.copy()


# Backward-compatible aliases.
FairLassoSelector = FairLassoStyleSelector
FairLassoBaselineSelector = FairLassoStyleSelector
FairLasso = FairLassoStyleSelector
