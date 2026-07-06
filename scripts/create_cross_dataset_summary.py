from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASET_PATHS = {
    "adult": "results/adult/final/adult_pilot_results.csv",
    "acs_income": "results/acs_income/final/acs_income_pilot_results.csv",
    "compas": "results/compas/final/compas_pilot_results.csv",
    "german_credit": "results/german_credit/final/german_credit_pilot_results.csv",
}


FAIRNESS_BASELINE_FAMILIES = {
    "ProxyRank",
    "fair-mRMR",
    "FairCFS-style",
    "FairLasso-style",
    "BasicFA-CMIM",
}


REQUIRED_COLUMNS = {
    "selector",
    "model",
    "accuracy",
    "balanced_accuracy",
    "dpd",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "joint_subset_mi_sensitive",
    "sensitive_attacker_balanced_accuracy",
    "selected_features",
}


def selector_family(selector: str) -> str:
    selector = str(selector)

    if selector.startswith("SubsetFACMIM"):
        return "SubsetFA-CMIM"

    if selector.startswith("BasicFACMIM"):
        return "BasicFA-CMIM"

    if selector.startswith("FairmRMR"):
        return "fair-mRMR"

    if selector.startswith("FairCFS"):
        return "FairCFS-style"

    if selector.startswith("FairLasso"):
        return "FairLasso-style"

    if selector.startswith("ProxyRank"):
        return "ProxyRank"

    if selector.startswith("CMIM"):
        return "CMIM"

    if selector.startswith("mRMR"):
        return "mRMR"

    return selector


def extract_k(selector: str) -> float:
    match = re.search(r"_k(\d+)", str(selector))
    return float(match.group(1)) if match else np.nan


def extract_lambda(selector: str) -> float:
    match = re.search(r"lambda([0-9.]+)", str(selector))
    return float(match.group(1)) if match else np.nan


def load_dataset_results(dataset_key: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_key} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dataset_key"] = dataset_key

    if "dataset" not in df.columns:
        df["dataset"] = dataset_key

    df["selector_family"] = df["selector"].apply(selector_family)
    df["k_value"] = df["selector"].apply(extract_k)
    df["lambda_value"] = df["selector"].apply(extract_lambda)

    return df


def load_all_results(dataset_paths: dict[str, str]) -> pd.DataFrame:
    frames = []

    for dataset_key, path_str in dataset_paths.items():
        path = Path(path_str)

        if not path.exists():
            print(f"Skipping missing dataset: {dataset_key} -> {path}")
            continue

        print(f"Loading {dataset_key}: {path}")
        frames.append(load_dataset_results(dataset_key, path))

    if not frames:
        raise FileNotFoundError("No dataset result files were found.")

    return pd.concat(frames, ignore_index=True)


def normalize_high_good(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    min_v = values.min()
    max_v = values.max()

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.ones(len(values)), index=series.index)

    return (values - min_v) / (max_v - min_v)


def normalize_low_good(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    min_v = values.min()
    max_v = values.max()

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.ones(len(values)), index=series.index)

    return 1.0 - ((values - min_v) / (max_v - min_v))


def add_tradeoff_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pieces = []

    for _, group in df.groupby(["dataset_key", "model"], dropna=False):
        group = group.copy()

        group["accuracy_norm"] = normalize_high_good(group["accuracy"])
        group["joint_leakage_quality"] = normalize_low_good(group["joint_subset_mi_sensitive"])
        group["attacker_leakage_quality"] = normalize_low_good(
            group["sensitive_attacker_balanced_accuracy"]
        )
        group["dpd_quality"] = normalize_low_good(group["dpd"])
        group["eod_quality"] = normalize_low_good(group["equal_opportunity_difference"])
        group["eodd_quality"] = normalize_low_good(group["equalized_odds_difference"])

        group["tradeoff_accuracy_joint_mi"] = (
            0.5 * group["accuracy_norm"] + 0.5 * group["joint_leakage_quality"]
        )

        group["tradeoff_accuracy_attacker"] = (
            0.5 * group["accuracy_norm"] + 0.5 * group["attacker_leakage_quality"]
        )

        group["tradeoff_accuracy_dpd"] = (
            0.5 * group["accuracy_norm"] + 0.5 * group["dpd_quality"]
        )

        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def mark_pareto(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pareto_accuracy_joint_mi"] = False

    for _, group in df.groupby(["dataset_key", "model"], dropna=False):
        idxs = list(group.index)

        acc = pd.to_numeric(group["accuracy"], errors="coerce").to_numpy()
        leak = pd.to_numeric(group["joint_subset_mi_sensitive"], errors="coerce").to_numpy()

        pareto_flags = []

        for i in range(len(group)):
            dominated = False

            for j in range(len(group)):
                if i == j:
                    continue

                better_or_equal_accuracy = acc[j] >= acc[i]
                better_or_equal_leakage = leak[j] <= leak[i]
                strictly_better = (acc[j] > acc[i]) or (leak[j] < leak[i])

                if better_or_equal_accuracy and better_or_equal_leakage and strictly_better:
                    dominated = True
                    break

            pareto_flags.append(not dominated)

        df.loc[idxs, "pareto_accuracy_joint_mi"] = pareto_flags

    return df


def pick_best_row(
    group: pd.DataFrame,
    *,
    sort_cols: list[str],
    ascending: list[bool],
) -> pd.Series:
    return group.sort_values(sort_cols, ascending=ascending).iloc[0]


def create_best_by_dataset_model(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (dataset_key, model), group in df.groupby(["dataset_key", "model"], dropna=False):
        criteria = {
            "best_accuracy": (
                ["accuracy", "joint_subset_mi_sensitive"],
                [False, True],
            ),
            "lowest_joint_subset_mi": (
                ["joint_subset_mi_sensitive", "accuracy"],
                [True, False],
            ),
            "lowest_attacker_balanced_accuracy": (
                ["sensitive_attacker_balanced_accuracy", "accuracy"],
                [True, False],
            ),
            "best_accuracy_joint_mi_tradeoff": (
                ["tradeoff_accuracy_joint_mi", "accuracy"],
                [False, False],
            ),
            "best_accuracy_attacker_tradeoff": (
                ["tradeoff_accuracy_attacker", "accuracy"],
                [False, False],
            ),
            "lowest_dpd": (
                ["dpd", "accuracy"],
                [True, False],
            ),
            "lowest_equalized_odds_difference": (
                ["equalized_odds_difference", "accuracy"],
                [True, False],
            ),
        }

        for criterion, (sort_cols, ascending) in criteria.items():
            best = pick_best_row(group, sort_cols=sort_cols, ascending=ascending)

            rows.append(
                {
                    "dataset_key": dataset_key,
                    "model": model,
                    "criterion": criterion,
                    "selector": best["selector"],
                    "selector_family": best["selector_family"],
                    "accuracy": best["accuracy"],
                    "balanced_accuracy": best["balanced_accuracy"],
                    "dpd": best["dpd"],
                    "equal_opportunity_difference": best["equal_opportunity_difference"],
                    "equalized_odds_difference": best["equalized_odds_difference"],
                    "joint_subset_mi_sensitive": best["joint_subset_mi_sensitive"],
                    "sensitive_attacker_balanced_accuracy": best[
                        "sensitive_attacker_balanced_accuracy"
                    ],
                    "tradeoff_accuracy_joint_mi": best["tradeoff_accuracy_joint_mi"],
                    "tradeoff_accuracy_attacker": best["tradeoff_accuracy_attacker"],
                    "selected_features": best["selected_features"],
                }
            )

    return pd.DataFrame(rows)


def create_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset_key", "selector_family", "model"], dropna=False)
        .agg(
            n_configs=("selector", "count"),
            best_accuracy=("accuracy", "max"),
            mean_accuracy=("accuracy", "mean"),
            lowest_dpd=("dpd", "min"),
            lowest_equal_opportunity_difference=("equal_opportunity_difference", "min"),
            lowest_equalized_odds_difference=("equalized_odds_difference", "min"),
            lowest_joint_subset_mi_sensitive=("joint_subset_mi_sensitive", "min"),
            mean_joint_subset_mi_sensitive=("joint_subset_mi_sensitive", "mean"),
            lowest_sensitive_attacker_balanced_accuracy=(
                "sensitive_attacker_balanced_accuracy",
                "min",
            ),
            mean_sensitive_attacker_balanced_accuracy=(
                "sensitive_attacker_balanced_accuracy",
                "mean",
            ),
            best_tradeoff_accuracy_joint_mi=("tradeoff_accuracy_joint_mi", "max"),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "dataset_key",
            "model",
            "lowest_joint_subset_mi_sensitive",
            "best_accuracy",
        ],
        ascending=[True, True, True, False],
    )


def best_proxy_row(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(
        [
            "joint_subset_mi_sensitive",
            "sensitive_attacker_balanced_accuracy",
            "accuracy",
        ],
        ascending=[True, True, False],
    ).iloc[0]


def create_subset_vs_baselines(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset_key, model), group in df.groupby(["dataset_key", "model"], dropna=False):
        subset_group = group[group["selector_family"] == "SubsetFA-CMIM"]
        cmim_group = group[group["selector_family"] == "CMIM"]
        baseline_group = group[group["selector_family"].isin(FAIRNESS_BASELINE_FAMILIES)]

        if subset_group.empty:
            continue

        subset_best = best_proxy_row(subset_group)

        cmim_best = best_proxy_row(cmim_group) if not cmim_group.empty else None
        baseline_best = best_proxy_row(baseline_group) if not baseline_group.empty else None

        row = {
            "dataset_key": dataset_key,
            "model": model,
            "subset_selector": subset_best["selector"],
            "subset_accuracy": subset_best["accuracy"],
            "subset_joint_mi": subset_best["joint_subset_mi_sensitive"],
            "subset_attacker_ba": subset_best["sensitive_attacker_balanced_accuracy"],
        }

        if cmim_best is not None:
            row.update(
                {
                    "cmim_selector": cmim_best["selector"],
                    "cmim_accuracy": cmim_best["accuracy"],
                    "cmim_joint_mi": cmim_best["joint_subset_mi_sensitive"],
                    "cmim_attacker_ba": cmim_best["sensitive_attacker_balanced_accuracy"],
                    "subset_joint_mi_advantage_vs_cmim": cmim_best[
                        "joint_subset_mi_sensitive"
                    ]
                    - subset_best["joint_subset_mi_sensitive"],
                    "subset_accuracy_delta_vs_cmim": subset_best["accuracy"]
                    - cmim_best["accuracy"],
                }
            )

        if baseline_best is not None:
            row.update(
                {
                    "best_fairness_baseline_selector": baseline_best["selector"],
                    "best_fairness_baseline_family": baseline_best["selector_family"],
                    "baseline_accuracy": baseline_best["accuracy"],
                    "baseline_joint_mi": baseline_best["joint_subset_mi_sensitive"],
                    "baseline_attacker_ba": baseline_best[
                        "sensitive_attacker_balanced_accuracy"
                    ],
                    "subset_joint_mi_advantage_vs_best_fairness_baseline": baseline_best[
                        "joint_subset_mi_sensitive"
                    ]
                    - subset_best["joint_subset_mi_sensitive"],
                    "subset_accuracy_delta_vs_best_fairness_baseline": subset_best[
                        "accuracy"
                    ]
                    - baseline_best["accuracy"],
                }
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["supports_subset_proxy_advantage_vs_cmim"] = (
        result["subset_joint_mi_advantage_vs_cmim"] > 1e-9
    )

    result["supports_subset_proxy_advantage_vs_fairness_baseline"] = (
        result["subset_joint_mi_advantage_vs_best_fairness_baseline"] > 1e-9
    )

    return result.sort_values(["dataset_key", "model"])


def create_claim_summary_by_dataset(subset_vs_baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset_key, group in subset_vs_baselines.groupby("dataset_key", dropna=False):
        n_models = len(group)

        n_vs_cmim = int(group["supports_subset_proxy_advantage_vs_cmim"].sum())
        n_vs_baseline = int(
            group["supports_subset_proxy_advantage_vs_fairness_baseline"].sum()
        )

        mean_adv_cmim = float(group["subset_joint_mi_advantage_vs_cmim"].mean())
        mean_adv_baseline = float(
            group["subset_joint_mi_advantage_vs_best_fairness_baseline"].mean()
        )

        if n_vs_baseline >= 2:
            interpretation = "strong_support_for_subset_proxy_leakage_advantage"
        elif n_vs_baseline == 1:
            interpretation = "partial_support_for_subset_proxy_leakage_advantage"
        else:
            interpretation = "flat_or_no_subset_proxy_leakage_advantage"

        rows.append(
            {
                "dataset_key": dataset_key,
                "n_models": n_models,
                "models_where_subset_beats_cmim_on_joint_mi": n_vs_cmim,
                "models_where_subset_beats_best_fairness_baseline_on_joint_mi": n_vs_baseline,
                "mean_subset_joint_mi_advantage_vs_cmim": mean_adv_cmim,
                "mean_subset_joint_mi_advantage_vs_best_fairness_baseline": mean_adv_baseline,
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows).sort_values("dataset_key")


def save_interpretation_notes(
    *,
    output_dir: Path,
    claim_summary: pd.DataFrame,
    subset_vs_baselines: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Cross-dataset interpretation notes")
    lines.append("")
    lines.append("Primary proxy-leakage metric: `joint_subset_mi_sensitive`.")
    lines.append("")
    lines.append("Lower values indicate less sensitive information retained by the selected feature subset.")
    lines.append("")
    lines.append("## Dataset-level claim summary")
    lines.append("")

    for _, row in claim_summary.iterrows():
        dataset_key = row["dataset_key"]
        interpretation = row["interpretation"]

        lines.append(f"### {dataset_key}")
        lines.append("")
        lines.append(
            f"- Models where Subset-aware FA-CMIM beats CMIM on joint MI: "
            f"{int(row['models_where_subset_beats_cmim_on_joint_mi'])} / "
            f"{int(row['n_models'])}"
        )
        lines.append(
            f"- Models where Subset-aware FA-CMIM beats the best fairness-aware baseline "
            f"on joint MI: "
            f"{int(row['models_where_subset_beats_best_fairness_baseline_on_joint_mi'])} / "
            f"{int(row['n_models'])}"
        )
        lines.append(
            f"- Mean joint-MI advantage versus CMIM: "
            f"{row['mean_subset_joint_mi_advantage_vs_cmim']:.6f}"
        )
        lines.append(
            f"- Mean joint-MI advantage versus best fairness-aware baseline: "
            f"{row['mean_subset_joint_mi_advantage_vs_best_fairness_baseline']:.6f}"
        )
        lines.append(f"- Interpretation: `{interpretation}`")
        lines.append("")

    lines.append("## Safe manuscript-level conclusion")
    lines.append("")
    lines.append(
        "The results should be interpreted as evidence that fairness-aware feature "
        "selection does not necessarily eliminate proxy leakage. Subset-aware "
        "FA-CMIM directly targets joint subset leakage and may provide stronger "
        "proxy mitigation when selectors have meaningful room to choose among "
        "competing features. However, in datasets where k is large relative to the "
        "available feature space, selectors can converge to similar subsets, which "
        "mutes cross-method leakage differences."
    )
    lines.append("")

    notes_path = output_dir / "cross_dataset_interpretation_notes.md"
    notes_path.write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy_vs_leakage(df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for family, group in df.groupby("selector_family"):
        ax.scatter(
            group["joint_subset_mi_sensitive"],
            group["accuracy"],
            label=family,
            alpha=0.75,
        )

    ax.set_xlabel("Joint subset MI with sensitive attribute")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs proxy leakage across datasets and models")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "cross_dataset_accuracy_vs_joint_mi.png", dpi=200)
    plt.close(fig)


def plot_subset_advantage(subset_vs_baselines: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if subset_vs_baselines.empty:
        return

    plot_df = subset_vs_baselines.copy()
    plot_df["dataset_model"] = plot_df["dataset_key"] + " | " + plot_df["model"]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        plot_df["dataset_model"],
        plot_df["subset_joint_mi_advantage_vs_best_fairness_baseline"],
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Joint-MI advantage of Subset-aware FA-CMIM vs best fairness baseline")
    ax.set_ylabel("Dataset | model")
    ax.set_title("Subset-aware FA-CMIM proxy-leakage advantage")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "subset_vs_fairness_baseline_joint_mi_advantage.png", dpi=200)
    plt.close(fig)


def run_cross_dataset_summary(args) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = {
        "adult": args.adult_path,
        "acs_income": args.acs_income_path,
        "compas": args.compas_path,
        "german_credit": args.german_credit_path,
    }

    all_results = load_all_results(dataset_paths)
    all_results = add_tradeoff_scores(all_results)
    all_results = mark_pareto(all_results)

    family_summary = create_family_summary(all_results)
    best_by_dataset_model = create_best_by_dataset_model(all_results)
    subset_vs_baselines = create_subset_vs_baselines(all_results)
    claim_summary = create_claim_summary_by_dataset(subset_vs_baselines)

    pareto = all_results[all_results["pareto_accuracy_joint_mi"] == True].copy()

    paths = {
        "all_results": output_dir / "cross_dataset_all_results.csv",
        "family_summary": output_dir / "cross_dataset_selector_family_summary.csv",
        "best_by_dataset_model": output_dir / "cross_dataset_best_by_dataset_model.csv",
        "pareto": output_dir / "cross_dataset_pareto_accuracy_joint_mi.csv",
        "subset_vs_baselines": output_dir / "cross_dataset_subset_vs_baselines.csv",
        "claim_summary": output_dir / "cross_dataset_claim_summary_by_dataset.csv",
        "interpretation_notes": output_dir / "cross_dataset_interpretation_notes.md",
    }

    all_results.to_csv(paths["all_results"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)
    best_by_dataset_model.to_csv(paths["best_by_dataset_model"], index=False)
    pareto.to_csv(paths["pareto"], index=False)
    subset_vs_baselines.to_csv(paths["subset_vs_baselines"], index=False)
    claim_summary.to_csv(paths["claim_summary"], index=False)

    save_interpretation_notes(
        output_dir=output_dir,
        claim_summary=claim_summary,
        subset_vs_baselines=subset_vs_baselines,
    )

    plot_accuracy_vs_leakage(all_results, output_dir)
    plot_subset_advantage(subset_vs_baselines, output_dir)

    print("\nSaved cross-dataset outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")

    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Create cross-dataset FA-CMIM summaries.")

    parser.add_argument("--adult_path", default=DEFAULT_DATASET_PATHS["adult"])
    parser.add_argument("--acs_income_path", default=DEFAULT_DATASET_PATHS["acs_income"])
    parser.add_argument("--compas_path", default=DEFAULT_DATASET_PATHS["compas"])
    parser.add_argument("--german_credit_path", default=DEFAULT_DATASET_PATHS["german_credit"])
    parser.add_argument("--output_dir", default="results/cross_dataset")

    return parser.parse_args()


def main():
    args = parse_args()
    run_cross_dataset_summary(args)


if __name__ == "__main__":
    main()
