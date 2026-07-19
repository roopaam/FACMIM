
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_adult_pilot import run_adult_pilot
from scripts.run_german_credit_pilot import (
    DEFAULT_GERMAN_URL,
    run_experiment as run_german_credit_experiment,
)


KEY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auroc",
    "dpd",
    "dpr",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "joint_subset_mi_sensitive",
    "sensitive_attacker_balanced_accuracy",
]


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def extract_lambda(selector_name: str) -> float:
    match = re.search(r"lambda([0-9.]+)", selector_name)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return np.nan
    return np.nan


def selector_family(selector_name: str) -> str:
    if selector_name.startswith("SubsetFACMIM"):
        return "Subset-aware FA-CMIM"
    if selector_name.startswith("BasicFACMIM"):
        return "Basic FA-CMIM"
    if selector_name.startswith("FairCFS"):
        return "FairCFS-style"
    if selector_name.startswith("FairLasso"):
        return "FairLasso-style"
    if selector_name.startswith("FairmRMR"):
        return "fair-mRMR"
    if selector_name.startswith("ProxyRank"):
        return "ProxyRank"
    if selector_name.startswith("CMIM"):
        return "CMIM"
    if selector_name.startswith("mRMR"):
        return "mRMR"
    return selector_name


def method_label(selector_name: str) -> str:
    family = selector_family(selector_name)
    lam = extract_lambda(selector_name)
    if not np.isnan(lam):
        return f"{family}, lambda={lam:g}"
    return family


def add_common_columns(df: pd.DataFrame, *, dataset: str, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset
    df["seed"] = seed
    df["selector_family"] = df["selector"].astype(str).map(selector_family)
    df["selector_lambda"] = df["selector"].astype(str).map(extract_lambda)
    df["method_label"] = df["selector"].astype(str).map(method_label)
    return df


def run_adult_seed(
    *,
    seed: int,
    output_root: Path,
    sample_size: int,
    k: int,
    lambdas: list[float],
    models: list[str],
    synthetic_smoke: bool,
) -> pd.DataFrame:
    out_dir = output_root / "raw" / "adult" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"Running Adult seed={seed}")
    print("=" * 90)

    df = run_adult_pilot(
        sample_size=sample_size,
        k=k,
        lambdas=lambdas,
        model_names=models,
        output_dir=out_dir,
        random_state=seed,
        synthetic_smoke=synthetic_smoke,
    )

    return add_common_columns(df, dataset="Adult", seed=seed)


def run_german_seed(
    *,
    seed: int,
    output_root: Path,
    sample_size: int,
    k: int,
    lambdas: list[float],
    models: list[str],
    test_size: float,
    data_path: str | None,
    download_url: str,
    age_threshold: int,
    synthetic_smoke: bool,
) -> pd.DataFrame:
    out_dir = output_root / "raw" / "german_credit" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"Running German Credit seed={seed}")
    print("=" * 90)

    args = SimpleNamespace(
        sample_size=sample_size,
        k=k,
        lambdas=lambdas,
        models=models,
        output_dir=str(out_dir),
        random_state=seed,
        test_size=test_size,
        data_path=data_path,
        download_url=download_url,
        age_threshold=age_threshold,
        synthetic_smoke=synthetic_smoke,
    )

    df = run_german_credit_experiment(args)

    return add_common_columns(df, dataset="German Credit", seed=seed)


def summarize_mean_std(full: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    available_metrics = [m for m in KEY_METRICS if m in full.columns]

    group_cols = [
        "dataset",
        "model",
        "selector",
        "selector_family",
        "selector_lambda",
        "method_label",
    ]

    summary = (
        full.groupby(group_cols, dropna=False)[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    summary_path = output_root / "adult_german_5seed_summary_all_methods.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved all-method mean/std summary to: {summary_path}")

    return summary


def pick_key_rows(summary: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []

    for dataset in ["Adult", "German Credit"]:
        for model in sorted(summary["model"].dropna().unique()):
            sub = summary[
                (summary["dataset"] == dataset)
                & (summary["model"] == model)
            ].copy()

            if sub.empty:
                continue

            # CMIM
            cmim = sub[sub["selector_family"] == "CMIM"]
            if not cmim.empty:
                rows.append(cmim.iloc[0].to_dict())

            # Strongest non-subset fairness-aware baseline by lowest mean joint MI.
            fairness_families = [
                "ProxyRank",
                "fair-mRMR",
                "FairCFS-style",
                "FairLasso-style",
                "Basic FA-CMIM",
            ]

            fairness = sub[sub["selector_family"].isin(fairness_families)].copy()

            # Prefer actual penalized versions when available.
            fairness_penalized = fairness[fairness["selector_lambda"].fillna(0) > 0]
            if not fairness_penalized.empty:
                fairness = fairness_penalized

            if "joint_subset_mi_sensitive_mean" in fairness.columns and not fairness.empty:
                fairness = fairness.sort_values(
                    ["joint_subset_mi_sensitive_mean", "accuracy_mean"],
                    ascending=[True, False],
                )
                best = fairness.iloc[0].to_dict()
                best["method_label"] = "Strongest fairness-aware baseline: " + str(best["method_label"])
                rows.append(best)

            # Subset-aware FA-CMIM lambda=1.0 and lambda=2.0
            for lam in [1.0, 2.0]:
                subset = sub[
                    (sub["selector_family"] == "Subset-aware FA-CMIM")
                    & (np.isclose(sub["selector_lambda"].astype(float), lam))
                ]
                if not subset.empty:
                    rows.append(subset.iloc[0].to_dict())

    key = pd.DataFrame(rows)

    key_path = output_root / "adult_german_5seed_key_table.csv"
    key.to_csv(key_path, index=False)
    print(f"Saved key robustness table to: {key_path}")

    return key


def fmt_mean_std(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in row:
        return "--"

    mean = safe_float(row.get(mean_col))
    std = safe_float(row.get(std_col))

    if np.isnan(mean):
        return "--"

    if np.isnan(std):
        std = 0.0

    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex_key_table(key: pd.DataFrame, output_root: Path) -> None:
    if key.empty:
        print("No key rows available for LaTeX table.")
        return

    latex_rows = []

    for _, row in key.iterrows():
        latex_rows.append(
            {
                "Dataset": row["dataset"],
                "Model": row["model"].replace("_", " ").title(),
                "Method": row["method_label"],
                "Accuracy": fmt_mean_std(row, "accuracy"),
                "Joint MI": fmt_mean_std(row, "joint_subset_mi_sensitive"),
                "Attacker BA": fmt_mean_std(row, "sensitive_attacker_balanced_accuracy"),
                "DPD": fmt_mean_std(row, "dpd"),
                "EODds": fmt_mean_std(row, "equalized_odds_difference"),
            }
        )

    table = pd.DataFrame(latex_rows)

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="lllccccc",
        caption=(
            "Five-seed robustness analysis for Adult and German Credit. "
            "Values are reported as mean $\\pm$ standard deviation across random seeds. "
            "Lower joint MI and lower attacker balanced accuracy indicate lower proxy leakage."
        ),
        label="tab:seed-robustness-main",
    )

    latex_path = output_root / "adult_german_5seed_key_table.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX key table to: {latex_path}")


def write_paired_seed_differences(full: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []

    try:
        from scipy.stats import wilcoxon
        scipy_available = True
    except Exception:
        wilcoxon = None
        scipy_available = False

    for dataset in ["Adult", "German Credit"]:
        for model in sorted(full["model"].dropna().unique()):
            sub = full[(full["dataset"] == dataset) & (full["model"] == model)].copy()

            cmim = sub[sub["selector_family"] == "CMIM"]
            if cmim.empty:
                continue

            cmim_by_seed = cmim.set_index("seed")["joint_subset_mi_sensitive"]

            for lam in [1.0, 2.0]:
                subset = sub[
                    (sub["selector_family"] == "Subset-aware FA-CMIM")
                    & (np.isclose(sub["selector_lambda"].astype(float), lam))
                ]

                if subset.empty:
                    continue

                subset_by_seed = subset.set_index("seed")["joint_subset_mi_sensitive"]

                common_seeds = sorted(set(cmim_by_seed.index) & set(subset_by_seed.index))

                if not common_seeds:
                    continue

                diffs = [
                    float(cmim_by_seed.loc[s] - subset_by_seed.loc[s])
                    for s in common_seeds
                ]

                p_value = np.nan
                if scipy_available and len(diffs) >= 2:
                    try:
                        stat = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
                        p_value = float(stat.pvalue)
                    except Exception:
                        p_value = np.nan

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "comparison": f"CMIM - Subset-aware FA-CMIM lambda={lam:g}",
                        "n_seeds": len(common_seeds),
                        "mean_delta_joint_mi": float(np.mean(diffs)),
                        "std_delta_joint_mi": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                        "wilcoxon_p_value_one_sided": p_value,
                        "note": "Positive delta means lower leakage for Subset-aware FA-CMIM.",
                    }
                )

    paired = pd.DataFrame(rows)
    path = output_root / "adult_german_5seed_paired_joint_mi_differences.csv"
    paired.to_csv(path, index=False)
    print(f"Saved paired seed-level differences to: {path}")

    return paired


def write_report(
    *,
    output_root: Path,
    args: argparse.Namespace,
    full: pd.DataFrame,
    summary: pd.DataFrame,
    key: pd.DataFrame,
    paired: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    report_path = output_root / "adult_german_5seed_report.md"

    lines = []
    lines.append("# Five-seed robustness analysis")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Seeds: `{args.seeds}`")
    lines.append(f"- Models: `{args.models}`")
    lines.append(f"- Feature budget k: `{args.k}`")
    lines.append(f"- Lambdas: `{args.lambdas}`")
    lines.append(f"- Adult sample size: `{args.adult_sample_size}`")
    lines.append(f"- German Credit sample size: `{args.german_sample_size}`")
    lines.append(f"- Runtime seconds: `{elapsed_seconds:.1f}`")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `adult_german_5seed_full_results.csv`")
    lines.append("- `adult_german_5seed_summary_all_methods.csv`")
    lines.append("- `adult_german_5seed_key_table.csv`")
    lines.append("- `adult_german_5seed_key_table.tex`")
    lines.append("- `adult_german_5seed_paired_joint_mi_differences.csv`")
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append(
        "The robustness analysis is intended as a stability check for the main leakage-reduction "
        "claims. Positive paired differences in joint MI indicate lower leakage for Subset-aware FA-CMIM "
        "relative to CMIM under the same random seed."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five-seed robustness analysis for Adult and German Credit."
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds to use.",
    )

    parser.add_argument("--k", type=int, default=8)

    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "gradient_boosting"],
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
    )

    parser.add_argument("--adult_sample_size", type=int, default=5000)
    parser.add_argument("--german_sample_size", type=int, default=1000)

    parser.add_argument("--output_dir", type=str, default="results/seed_robustness")

    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--german_data_path", type=str, default=None)
    parser.add_argument("--german_download_url", type=str, default=DEFAULT_GERMAN_URL)
    parser.add_argument("--german_age_threshold", type=int, default=25)

    parser.add_argument(
        "--adult_synthetic_smoke",
        action="store_true",
        help="Use synthetic Adult-like data for smoke testing only.",
    )

    parser.add_argument(
        "--german_synthetic_smoke",
        action="store_true",
        help="Use synthetic German-like data for smoke testing only.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.time()

    all_results = []

    config_path = output_root / "adult_german_5seed_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    for seed in args.seeds:
        adult = run_adult_seed(
            seed=seed,
            output_root=output_root,
            sample_size=args.adult_sample_size,
            k=args.k,
            lambdas=args.lambdas,
            models=args.models,
            synthetic_smoke=args.adult_synthetic_smoke,
        )
        all_results.append(adult)

        german = run_german_seed(
            seed=seed,
            output_root=output_root,
            sample_size=args.german_sample_size,
            k=args.k,
            lambdas=args.lambdas,
            models=args.models,
            test_size=args.test_size,
            data_path=args.german_data_path,
            download_url=args.german_download_url,
            age_threshold=args.german_age_threshold,
            synthetic_smoke=args.german_synthetic_smoke,
        )
        all_results.append(german)

    full = pd.concat(all_results, ignore_index=True)

    full_path = output_root / "adult_german_5seed_full_results.csv"
    full.to_csv(full_path, index=False)
    print(f"\nSaved full results to: {full_path}")

    summary = summarize_mean_std(full, output_root)
    key = pick_key_rows(summary, output_root)
    write_latex_key_table(key, output_root)
    paired = write_paired_seed_differences(full, output_root)

    elapsed = time.time() - start

    write_report(
        output_root=output_root,
        args=args,
        full=full,
        summary=summary,
        key=key,
        paired=paired,
        elapsed_seconds=elapsed,
    )

    print("\n" + "=" * 90)
    print("Five-seed robustness analysis complete.")
    print("=" * 90)
    print(f"Output directory: {output_root}")
    print(f"Runtime: {elapsed:.1f} seconds")

    key_display_cols = [
        "dataset",
        "model",
        "method_label",
        "accuracy_mean",
        "accuracy_std",
        "joint_subset_mi_sensitive_mean",
        "joint_subset_mi_sensitive_std",
        "sensitive_attacker_balanced_accuracy_mean",
        "sensitive_attacker_balanced_accuracy_std",
    ]

    key_display_cols = [c for c in key_display_cols if c in key.columns]

    if not key.empty:
        print("\nKey robustness rows:")
        print(key[key_display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
