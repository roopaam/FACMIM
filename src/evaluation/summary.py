from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auroc",
    "dpd",
    "dpr",
    "equal_opportunity_difference",
    "equal_opportunity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
    "mean_selected_mi_sensitive",
    "max_selected_mi_sensitive",
    "joint_subset_mi_sensitive",
    "sensitive_attacker_balanced_accuracy",
    "selected_feature_count",
]


def load_results(results_path: str | Path) -> pd.DataFrame:
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    if results_path.suffix.lower() == ".csv":
        return pd.read_csv(results_path)

    if results_path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(results_path, lines=results_path.suffix.lower() == ".jsonl")

    raise ValueError(f"Unsupported results file type: {results_path.suffix}")


def _numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    cols = []

    for col in DEFAULT_METRIC_COLUMNS:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors="raise")
                cols.append(col)
            except Exception:
                pass

    return cols


def _safe_min_max_norm(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")

    mn = values.min(skipna=True)
    mx = values.max(skipna=True)

    if pd.isna(mn) or pd.isna(mx):
        return pd.Series(np.nan, index=values.index)

    if mx == mn:
        return pd.Series(1.0, index=values.index)

    norm = (values - mn) / (mx - mn)

    if higher_is_better:
        return norm

    return 1.0 - norm


def add_tradeoff_scores(
    df: pd.DataFrame,
    *,
    utility_col: str = "accuracy",
    fairness_col: str = "dpd",
    group_col: str = "model",
    utility_weight: float = 0.5,
    fairness_weight: float = 0.5,
) -> pd.DataFrame:
    out = df.copy()

    if utility_col not in out.columns:
        raise ValueError(f"Missing utility column: {utility_col}")

    if fairness_col not in out.columns:
        raise ValueError(f"Missing fairness column: {fairness_col}")

    out["utility_norm"] = np.nan
    out["fairness_norm"] = np.nan
    out["tradeoff_score"] = np.nan

    if group_col in out.columns:
        groups = out.groupby(group_col, dropna=False)
    else:
        groups = [(None, out)]

    for _, group in groups:
        utility_norm = _safe_min_max_norm(group[utility_col], higher_is_better=True)
        fairness_norm = _safe_min_max_norm(group[fairness_col], higher_is_better=False)

        out.loc[group.index, "utility_norm"] = utility_norm
        out.loc[group.index, "fairness_norm"] = fairness_norm
        out.loc[group.index, "tradeoff_score"] = (
            utility_weight * utility_norm + fairness_weight * fairness_norm
        )

    return out


def compute_pareto_frontier(
    df: pd.DataFrame,
    *,
    utility_col: str = "accuracy",
    fairness_col: str = "dpd",
    group_cols: list[str] | tuple[str, ...] | None = ("model",),
) -> pd.DataFrame:
    """
    Mark Pareto-optimal rows.

    A row is dominated if another row in the same group has:
        - utility >= this utility
        - fairness <= this fairness
        - at least one strict improvement

    Higher utility is better. Lower fairness gap/leakage is better.
    """
    out = df.copy()

    if utility_col not in out.columns:
        raise ValueError(f"Missing utility column: {utility_col}")

    if fairness_col not in out.columns:
        raise ValueError(f"Missing fairness column: {fairness_col}")

    out["pareto_optimal"] = False
    out["pareto_utility_col"] = utility_col
    out["pareto_fairness_col"] = fairness_col

    if group_cols:
        group_cols = [c for c in group_cols if c in out.columns]

    if group_cols:
        groups = out.groupby(group_cols, dropna=False)
    else:
        groups = [(None, out)]

    for _, group in groups:
        g = group.copy()
        g[utility_col] = pd.to_numeric(g[utility_col], errors="coerce")
        g[fairness_col] = pd.to_numeric(g[fairness_col], errors="coerce")
        g = g.dropna(subset=[utility_col, fairness_col])

        if g.empty:
            continue

        for idx, row in g.iterrows():
            u = row[utility_col]
            f = row[fairness_col]

            dominates = (
                (g[utility_col] >= u)
                & (g[fairness_col] <= f)
                & ((g[utility_col] > u) | (g[fairness_col] < f))
            )

            out.loc[idx, "pareto_optimal"] = not bool(dominates.any())

    return out


def selector_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["selector", "model"] if c in df.columns]

    if not group_cols:
        raise ValueError("Expected at least one grouping column: selector or model")

    metric_cols = _numeric_metric_columns(df)

    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    if "selected_features" in df.columns:
        selected = (
            df.groupby(group_cols, dropna=False)["selected_features"]
            .first()
            .reset_index()
        )
        summary = summary.merge(selected, on=group_cols, how="left")

    return summary


def best_rows_by_group(
    df: pd.DataFrame,
    *,
    metric_col: str,
    group_col: str = "model",
    maximize: bool = True,
) -> pd.DataFrame:
    if metric_col not in df.columns:
        raise ValueError(f"Missing metric column: {metric_col}")

    valid = df.copy()
    valid[metric_col] = pd.to_numeric(valid[metric_col], errors="coerce")
    valid = valid.dropna(subset=[metric_col])

    if valid.empty:
        return valid

    ascending_metric = not maximize

    if group_col in valid.columns:
        valid = valid.sort_values([group_col, metric_col], ascending=[True, ascending_metric])
        return valid.groupby(group_col, dropna=False).head(1).reset_index(drop=True)

    valid = valid.sort_values(metric_col, ascending=ascending_metric)
    return valid.head(1).reset_index(drop=True)


def save_summary_outputs(
    results_path: str | Path,
    output_dir: str | Path,
    *,
    utility_col: str = "accuracy",
    fairness_col: str = "dpd",
) -> dict[str, str]:
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    scored = add_tradeoff_scores(
        results,
        utility_col=utility_col,
        fairness_col=fairness_col,
    )

    summary = selector_model_summary(scored)
    pareto = compute_pareto_frontier(
        scored,
        utility_col=utility_col,
        fairness_col=fairness_col,
        group_cols=("model",),
    )

    best_accuracy = best_rows_by_group(
        scored,
        metric_col=utility_col,
        group_col="model",
        maximize=True,
    )

    best_fairness = best_rows_by_group(
        scored,
        metric_col=fairness_col,
        group_col="model",
        maximize=False,
    )

    best_tradeoff = best_rows_by_group(
        scored,
        metric_col="tradeoff_score",
        group_col="model",
        maximize=True,
    )

    paths = {
        "scored_results": str(output_dir / "scored_results.csv"),
        "selector_model_summary": str(output_dir / "selector_model_summary.csv"),
        "pareto": str(output_dir / f"pareto_{utility_col}_{fairness_col}.csv"),
        "best_accuracy_by_model": str(output_dir / "best_accuracy_by_model.csv"),
        "best_fairness_by_model": str(output_dir / "best_fairness_by_model.csv"),
        "best_tradeoff_by_model": str(output_dir / "best_tradeoff_by_model.csv"),
        "metadata": str(output_dir / "summary_metadata.json"),
    }

    scored.to_csv(paths["scored_results"], index=False)
    summary.to_csv(paths["selector_model_summary"], index=False)
    pareto.to_csv(paths["pareto"], index=False)
    best_accuracy.to_csv(paths["best_accuracy_by_model"], index=False)
    best_fairness.to_csv(paths["best_fairness_by_model"], index=False)
    best_tradeoff.to_csv(paths["best_tradeoff_by_model"], index=False)

    metadata = {
        "results_path": str(results_path),
        "output_dir": str(output_dir),
        "row_count": int(len(results)),
        "columns": list(results.columns),
        "utility_col": utility_col,
        "fairness_col": fairness_col,
        "outputs": paths,
    }

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return paths


def _sanitize_filename(value: Any) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "all"


def plot_pareto_frontier(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    utility_col: str = "accuracy",
    fairness_col: str = "dpd",
    label_col: str = "selector",
    title: str | None = None,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = compute_pareto_frontier(
        df,
        utility_col=utility_col,
        fairness_col=fairness_col,
        group_cols=None,
    )

    plot_df[utility_col] = pd.to_numeric(plot_df[utility_col], errors="coerce")
    plot_df[fairness_col] = pd.to_numeric(plot_df[fairness_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[utility_col, fairness_col])

    if plot_df.empty:
        raise ValueError("No valid rows available for Pareto plotting.")

    pareto = plot_df[plot_df["pareto_optimal"]]
    non_pareto = plot_df[~plot_df["pareto_optimal"]]

    plt.figure(figsize=(8, 6))

    if not non_pareto.empty:
        plt.scatter(
            non_pareto[fairness_col],
            non_pareto[utility_col],
            label="Non-Pareto",
            alpha=0.75,
        )

    if not pareto.empty:
        plt.scatter(
            pareto[fairness_col],
            pareto[utility_col],
            label="Pareto-optimal",
            marker="x",
            s=80,
        )

    if label_col in pareto.columns:
        for _, row in pareto.iterrows():
            label = str(row[label_col])[:40]
            plt.annotate(
                label,
                (row[fairness_col], row[utility_col]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    plt.xlabel(f"{fairness_col} lower is better")
    plt.ylabel(f"{utility_col} higher is better")

    if title is None:
        title = f"Pareto frontier: {utility_col} vs {fairness_col}"

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return str(output_path)


def plot_all_pareto(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    utility_col: str = "accuracy",
    fairness_cols: list[str] | tuple[str, ...] = (
        "dpd",
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "joint_subset_mi_sensitive",
    ),
    group_col: str = "model",
    label_col: str = "selector",
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    if group_col in df.columns:
        group_values = list(df[group_col].dropna().unique())
    else:
        group_values = ["all"]

    for fairness_col in fairness_cols:
        if fairness_col not in df.columns:
            continue

        for group_value in group_values:
            if group_col in df.columns:
                sub = df[df[group_col] == group_value].copy()
            else:
                sub = df.copy()

            if sub.empty:
                continue

            filename = (
                f"pareto_{utility_col}_vs_{fairness_col}_"
                f"{_sanitize_filename(group_value)}.png"
            )

            title = f"{group_value}: {utility_col} vs {fairness_col}"

            path = plot_pareto_frontier(
                sub,
                output_dir / filename,
                utility_col=utility_col,
                fairness_col=fairness_col,
                label_col=label_col,
                title=title,
            )

            saved_paths.append(path)

    return saved_paths
