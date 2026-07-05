from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def _to_series(x: Any, index=None, name: str | None = None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)

    if index is not None:
        s = pd.Series(s.to_numpy(), index=index)

    if name is not None:
        s.name = name

    return s


def _discretize_series_local(s: pd.Series, n_bins: int = 5) -> pd.Series:
    """
    Discretize a series for MI/CMI calculations.

    Low-cardinality numeric variables, including binary 0/1 columns, are
    treated as categorical values directly. This avoids qcut collapsing them
    into a single bin.
    """
    s = pd.Series(s).copy()

    non_missing = s.dropna()
    nunique = non_missing.nunique(dropna=True)

    if nunique <= 1:
        return s.astype("object").where(s.notna(), "__missing__").astype(str)

    if pd.api.types.is_numeric_dtype(s):
        if nunique <= n_bins:
            return s.astype("object").where(s.notna(), "__missing__").astype(str)

        try:
            out = pd.qcut(s, q=n_bins, duplicates="drop")
            return out.astype("object").where(pd.notna(out), "__missing__").astype(str)
        except Exception:
            try:
                out = pd.cut(s, bins=n_bins, duplicates="drop")
                return out.astype("object").where(pd.notna(out), "__missing__").astype(str)
            except Exception:
                return s.astype("object").where(s.notna(), "__missing__").astype(str)

    return s.astype("object").where(s.notna(), "__missing__").astype(str)


def _mi(x: pd.Series, y: pd.Series) -> float:
    x = _discretize_series_local(pd.Series(x))
    y = _discretize_series_local(pd.Series(y))
    return float(mutual_info_score(x, y))


def _cmi(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """
    Estimate I(X;Y|Z) using:

        I(X;Y|Z) = sum_z p(z) I(X;Y | Z=z)
    """
    x = _discretize_series_local(pd.Series(x))
    y = _discretize_series_local(pd.Series(y))
    z = _discretize_series_local(pd.Series(z))

    n = len(z)
    if n == 0:
        return 0.0

    total = 0.0

    for value in z.unique():
        mask = z == value
        count = int(mask.sum())

        if count == 0:
            continue

        weight = count / n
        total += weight * mutual_info_score(x[mask], y[mask])

    return float(total)


# CMIM objective based on Fleuret (2004),
# "Fast Binary Feature Selection with Conditional Mutual Information."
class CMIMSelector:
    """
    Greedy Conditional Mutual Information Maximization selector.

    Objective:

        First selected feature:
            score_j = I(X_j ; Y)

        Subsequent selected features:
            score_j = min_{X_s in selected} I(X_j ; Y | X_s)

    At each iteration, select the candidate with the highest score.
    """

    def __init__(
        self,
        k: int = 10,
        *,
        n_features: int | None = None,
        n_features_to_select: int | None = None,
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

        if int(k) <= 0:
            raise ValueError("k must be positive.")

        if int(n_bins) < 2:
            raise ValueError("n_bins must be at least 2.")

        self.k = int(k)
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
    ) -> "CMIMSelector":
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

        selected: list[str] = []
        remaining = list(self.candidate_features_)
        diagnostics_rows: list[dict[str, Any]] = []

        marginal_relevance = {c: _mi(Xd[c], yd) for c in remaining}
        n_to_select = min(self.k, len(remaining))

        for rank in range(1, n_to_select + 1):
            scored_rows = []

            for c in remaining:
                if not selected:
                    cmim_score = marginal_relevance[c]
                else:
                    cmim_score = min(_cmi(Xd[c], yd, Xd[s]) for s in selected)

                score = float(cmim_score)

                scored_rows.append(
                    {
                        "candidate": c,
                        "marginal_relevance": float(marginal_relevance[c]),
                        "cmim_score": float(cmim_score),
                        "score": score,
                    }
                )

            order_index = {c: i for i, c in enumerate(self.candidate_features_)}

            scored_rows = sorted(
                scored_rows,
                key=lambda r: (
                    -r["score"],
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
            raise RuntimeError("CMIMSelector must be fitted before transform().")

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
            raise RuntimeError("CMIMSelector must be fitted before get_support().")

        return list(self.selected_features_)

    def get_selected_features(self) -> list[str]:
        return self.get_support()

    def get_support_mask(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("CMIMSelector must be fitted before get_support_mask().")

        return np.array(
            [c in self.selected_features_ for c in self.feature_names_in_],
            dtype=bool,
        )

    def get_diagnostics(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("CMIMSelector must be fitted before get_diagnostics().")

        return self.diagnostics_.copy()

    def get_feature_scores(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("CMIMSelector must be fitted before get_feature_scores().")

        return self.feature_scores_.copy()
