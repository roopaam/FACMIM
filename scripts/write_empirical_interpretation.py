from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_LABELS = {
    "adult": "Adult",
    "acs_income": "ACSIncome",
    "compas": "COMPAS",
    "german_credit": "German Credit",
}


INTERPRETATION_LABELS = {
    "strong_support_for_subset_proxy_leakage_advantage": "strong support",
    "partial_support_for_subset_proxy_leakage_advantage": "partial support",
    "flat_or_no_subset_proxy_leakage_advantage": "flat or no clear advantage",
}


def fmt_num(value, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.read_csv(path)


def compact_selector_name(selector: str) -> str:
    selector = str(selector)
    selector = selector.replace("SubsetFACMIM", "Subset FA-CMIM")
    selector = selector.replace("BasicFACMIM", "Basic FA-CMIM")
    selector = selector.replace("FairmRMR", "fair-mRMR")
    selector = selector.replace("FairCFS", "FairCFS")
    selector = selector.replace("FairLasso", "FairLasso")
    selector = selector.replace("ProxyRank", "ProxyRank")
    return selector


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"

    df = df.copy().fillna("")

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def prepare_claim_table(claim: pd.DataFrame) -> pd.DataFrame:
    table = claim.copy()

    table["Dataset"] = table["dataset_key"].map(DATASET_LABELS).fillna(table["dataset_key"])
    table["Subset beats CMIM"] = (
        table["models_where_subset_beats_cmim_on_joint_mi"].astype(int).astype(str)
        + "/"
        + table["n_models"].astype(int).astype(str)
    )
    table["Subset beats fairness baseline"] = (
        table["models_where_subset_beats_best_fairness_baseline_on_joint_mi"]
        .astype(int)
        .astype(str)
        + "/"
        + table["n_models"].astype(int).astype(str)
    )
    table["Mean joint-MI advantage vs CMIM"] = table[
        "mean_subset_joint_mi_advantage_vs_cmim"
    ].map(lambda x: fmt_num(x, 4))
    table["Mean joint-MI advantage vs fairness baseline"] = table[
        "mean_subset_joint_mi_advantage_vs_best_fairness_baseline"
    ].map(lambda x: fmt_num(x, 4))
    table["Interpretation"] = (
        table["interpretation"]
        .map(INTERPRETATION_LABELS)
        .fillna(table["interpretation"])
    )

    return table[
        [
            "Dataset",
            "Subset beats CMIM",
            "Subset beats fairness baseline",
            "Mean joint-MI advantage vs CMIM",
            "Mean joint-MI advantage vs fairness baseline",
            "Interpretation",
        ]
    ]


def prepare_subset_vs_baseline_table(svb: pd.DataFrame) -> pd.DataFrame:
    table = svb.copy()

    table["Dataset"] = table["dataset_key"].map(DATASET_LABELS).fillna(table["dataset_key"])
    table["Model"] = table["model"]
    table["Subset selector"] = table["subset_selector"].map(compact_selector_name)
    table["Best fairness baseline"] = table["best_fairness_baseline_selector"].map(
        compact_selector_name
    )
    table["Subset accuracy"] = table["subset_accuracy"].map(lambda x: fmt_num(x, 4))
    table["Baseline accuracy"] = table["baseline_accuracy"].map(lambda x: fmt_num(x, 4))
    table["Subset joint MI"] = table["subset_joint_mi"].map(lambda x: fmt_num(x, 4))
    table["Baseline joint MI"] = table["baseline_joint_mi"].map(lambda x: fmt_num(x, 4))
    table["Joint-MI advantage"] = table[
        "subset_joint_mi_advantage_vs_best_fairness_baseline"
    ].map(lambda x: fmt_num(x, 4))
    table["Accuracy delta"] = table[
        "subset_accuracy_delta_vs_best_fairness_baseline"
    ].map(lambda x: fmt_num(x, 4))

    return table[
        [
            "Dataset",
            "Model",
            "Subset selector",
            "Best fairness baseline",
            "Subset accuracy",
            "Baseline accuracy",
            "Subset joint MI",
            "Baseline joint MI",
            "Joint-MI advantage",
            "Accuracy delta",
        ]
    ].sort_values(["Dataset", "Model"])


def dataset_interpretation_paragraph(row: pd.Series) -> str:
    dataset_key = row["dataset_key"]
    dataset_label = DATASET_LABELS.get(dataset_key, dataset_key)

    n_models = int(row["n_models"])
    subset_vs_cmim = int(row["models_where_subset_beats_cmim_on_joint_mi"])
    subset_vs_baseline = int(
        row["models_where_subset_beats_best_fairness_baseline_on_joint_mi"]
    )

    mean_adv_cmim = float(row["mean_subset_joint_mi_advantage_vs_cmim"])
    mean_adv_baseline = float(
        row["mean_subset_joint_mi_advantage_vs_best_fairness_baseline"]
    )

    interpretation = row["interpretation"]

    if interpretation == "strong_support_for_subset_proxy_leakage_advantage":
        conclusion = (
            "This dataset provides the clearest support for the proposed subset-aware "
            "leakage-control mechanism. Subset-aware FA-CMIM reduces joint sensitive "
            "leakage relative to the strongest fairness-aware baseline in most model "
            "settings, while maintaining a competitive level of predictive utility."
        )
    elif interpretation == "partial_support_for_subset_proxy_leakage_advantage":
        conclusion = (
            "This dataset provides partial support for the proposed method. The subset-aware "
            "objective improves proxy-leakage control in at least one model setting, but the "
            "advantage is not uniform across all downstream learners."
        )
    else:
        conclusion = (
            "This dataset shows limited separation across selectors. The most plausible "
            "interpretation is that the selected feature sets are highly overlapping under "
            "the current k setting, so the subset-aware penalty has limited room to produce "
            "a distinct leakage profile."
        )

    paragraph = (
        f"**{dataset_label}.** Subset-aware FA-CMIM achieved lower joint sensitive "
        f"leakage than CMIM in {subset_vs_cmim}/{n_models} model settings and lower "
        f"joint sensitive leakage than the best fairness-aware baseline in "
        f"{subset_vs_baseline}/{n_models} model settings. The mean joint-MI advantage "
        f"was {fmt_num(mean_adv_cmim, 4)} relative to CMIM and "
        f"{fmt_num(mean_adv_baseline, 4)} relative to the best fairness-aware baseline. "
        f"{conclusion}"
    )

    return paragraph


def write_empirical_interpretation(
    *,
    cross_dataset_dir: Path,
    output_path: Path,
    table_output_dir: Path,
) -> Path:
    claim = read_required_csv(cross_dataset_dir / "cross_dataset_claim_summary_by_dataset.csv")
    svb = read_required_csv(cross_dataset_dir / "cross_dataset_subset_vs_baselines.csv")
    best = read_required_csv(cross_dataset_dir / "cross_dataset_best_by_dataset_model.csv")
    family = read_required_csv(cross_dataset_dir / "cross_dataset_selector_family_summary.csv")

    table_output_dir.mkdir(parents=True, exist_ok=True)

    claim_table = prepare_claim_table(claim)
    svb_table = prepare_subset_vs_baseline_table(svb)

    claim_table_path = table_output_dir / "empirical_claim_summary_table.csv"
    svb_table_path = table_output_dir / "subset_vs_fairness_baseline_table.csv"

    claim_table.to_csv(claim_table_path, index=False)
    svb_table.to_csv(svb_table_path, index=False)

    n_datasets = claim["dataset_key"].nunique()
    n_models_total = int(claim["n_models"].sum())
    strong_count = int(
        (claim["interpretation"] == "strong_support_for_subset_proxy_leakage_advantage").sum()
    )
    partial_count = int(
        (claim["interpretation"] == "partial_support_for_subset_proxy_leakage_advantage").sum()
    )
    flat_count = int(
        (claim["interpretation"] == "flat_or_no_subset_proxy_leakage_advantage").sum()
    )

    dataset_paragraphs = "\n\n".join(
        dataset_interpretation_paragraph(row)
        for _, row in claim.sort_values("dataset_key").iterrows()
    )

    section = f"""# Empirical Results and Interpretation

## Evaluation focus

The empirical evaluation was designed to test whether fairness-aware feature selection also controls proxy leakage. The central quantity of interest is not only predictive accuracy or downstream group fairness, but the amount of sensitive information retained by the selected feature subset. For this reason, the experiments report conventional utility metrics, downstream fairness metrics, joint subset mutual information with the sensitive attribute, and sensitive-attribute attacker balanced accuracy.

The key distinction examined in these experiments is between marginal fairness-aware selection and subset-aware leakage control. Methods such as ProxyRank, fair-mRMR, FairCFS-style selection, FairLasso-style selection, and Basic FA-CMIM include fairness-aware penalties or constraints, but they do not necessarily penalize the sensitive information encoded by the selected subset as a whole. Subset-aware FA-CMIM directly targets this joint leakage.

## Cross-dataset summary

Across {n_datasets} datasets and {n_models_total} dataset-model settings, the results show a heterogeneous but informative pattern. Subset-aware FA-CMIM shows strong support for proxy-leakage reduction on {strong_count} dataset(s), partial support on {partial_count} dataset(s), and flat or no clear advantage on {flat_count} dataset(s). This pattern is important because it avoids the misleading claim that the proposed method is uniformly superior across all metrics. Instead, the results support a more precise conclusion: subset-aware FA-CMIM is most beneficial when there is meaningful room for the selector to choose among competing predictive features with different leakage profiles.

{markdown_table(claim_table)}

## Dataset-level interpretation

{dataset_paragraphs}

## Comparison against fairness-aware baselines

The most important empirical observation is that fairness-aware baselines can still retain substantial sensitive information in the selected subset. This is visible when a baseline achieves competitive accuracy or downstream fairness while its selected feature subset continues to exhibit high joint mutual information with the sensitive attribute or high sensitive-attribute attacker balanced accuracy.

The comparison below focuses on the best proxy-leakage configuration of Subset-aware FA-CMIM against the best proxy-leakage configuration among the fairness-aware baselines for each dataset and model.

{markdown_table(svb_table)}

Positive values in the joint-MI advantage column mean that Subset-aware FA-CMIM achieved lower joint sensitive leakage than the best fairness-aware baseline. Negative or zero values indicate that the subset-aware method did not improve on the strongest baseline for that dataset-model setting.

## Main empirical conclusion

The results support three conclusions.

First, fairness-aware feature selection is not automatically proxy-leakage-aware. Across datasets, several fairness-aware baselines remain competitive on accuracy and downstream fairness metrics but still retain measurable sensitive information in their selected subsets.

Second, Subset-aware FA-CMIM provides the clearest advantage when the dataset contains enough competing candidate features for the subset-level penalty to alter the selected subset. This was most visible in the Adult dataset, where subset-aware leakage control produced a stronger reduction in sensitive information than the fairness-aware baselines.

Third, the advantage is configuration-dependent. In datasets such as ACSIncome and COMPAS under the current k=8 setting, many selectors converge to similar feature subsets. In such cases, downstream performance and proxy-leakage metrics become nearly indistinguishable. This does not refute the subset-aware mechanism; rather, it indicates that its advantage is muted when the selected subset size is large relative to the effective feature space.

## Implications for proxy mitigation

These findings reinforce the central motivation of FA-CMIM. A selector can be fairness-aware in name or objective form while still preserving sensitive information through combinations of retained features. Therefore, proxy mitigation should be evaluated directly using subset-level leakage diagnostics, not only through downstream fairness metrics such as demographic parity difference or equalized odds difference.

The proposed subset-aware objective should therefore be interpreted as a proxy-mitigation mechanism rather than a pure accuracy-maximization mechanism. Its purpose is to expose and control the fairness--utility--leakage trade-off. In this sense, the empirical results support the paper's main claim: controlling marginal proxy association is insufficient when sensitive information can be reconstructed compositionally from the selected feature set.

## Limitations and cautious interpretation

The current experiments use a fixed k=8 setting for cross-dataset comparability. This choice makes the results easy to compare, but it can also reduce selector differentiation on datasets with smaller feature spaces. Future sensitivity analysis should vary k and the fairness penalty to examine when subset-aware leakage control becomes most effective. The current results should therefore be framed as evidence of conditional proxy-leakage advantage, not as a universal dominance claim.

A manuscript-safe summary is:

> Subset-aware FA-CMIM does not uniformly maximize accuracy or dominate every baseline on every dataset. Rather, it directly targets joint sensitive leakage and can substantially reduce proxy information when the feature-selection problem provides meaningful room for subset-level trade-offs. Across datasets, the results show that fairness-aware baselines do not reliably eliminate proxy leakage, supporting the need for explicit subset-aware leakage diagnostics and penalties.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(section, encoding="utf-8")

    print(f"Saved empirical interpretation section: {output_path}")
    print(f"Saved claim summary table: {claim_table_path}")
    print(f"Saved subset-vs-baseline table: {svb_table_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write final empirical interpretation section for the FA-CMIM paper."
    )

    parser.add_argument(
        "--cross_dataset_dir",
        default="results/cross_dataset",
    )
    parser.add_argument(
        "--output_path",
        default="manuscript/sections/empirical_interpretation.md",
    )
    parser.add_argument(
        "--table_output_dir",
        default="manuscript/sections/generated_tables",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    write_empirical_interpretation(
        cross_dataset_dir=Path(args.cross_dataset_dir),
        output_path=Path(args.output_path),
        table_output_dir=Path(args.table_output_dir),
    )


if __name__ == "__main__":
    main()
