from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_LABELS = {
    "adult": "Adult",
    "acs_income": "ACSIncome",
    "compas": "COMPAS",
    "german_credit": "German Credit",
}


MODEL_LABELS = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "gradient_boosting": "GB",
}


INTERPRETATION_LABELS = {
    "strong_support_for_subset_proxy_leakage_advantage": "Strong support",
    "partial_support_for_subset_proxy_leakage_advantage": "Partial support",
    "flat_or_no_subset_proxy_leakage_advantage": "Flat / no clear advantage",
}


FAMILY_ORDER = [
    "CMIM",
    "mRMR",
    "ProxyRank",
    "fair-mRMR",
    "FairCFS-style",
    "FairLasso-style",
    "BasicFA-CMIM",
    "SubsetFA-CMIM",
]


def latex_escape(value) -> str:
    text = "" if pd.isna(value) else str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def fmt_num(value, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"
    except Exception:
        return latex_escape(value)


def safe_label(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def compact_selector(selector: str) -> str:
    selector = str(selector)

    selector = selector.replace("SubsetFACMIM", "Subset FA-CMIM")
    selector = selector.replace("BasicFACMIM", "Basic FA-CMIM")
    selector = selector.replace("FairmRMR", "fair-mRMR")
    selector = selector.replace("FairCFS", "FairCFS")
    selector = selector.replace("FairLasso", "FairLasso")
    selector = selector.replace("ProxyRank", "ProxyRank")

    selector = selector.replace("_lambda", r" $\lambda=$")
    selector = selector.replace("_k", r" $k=$")

    return selector


def dataset_label(dataset_key: str) -> str:
    return DATASET_LABELS.get(str(dataset_key), str(dataset_key))


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def make_latex_table(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    align: str | None = None,
    font_size: str = r"\scriptsize",
    resize: bool = True,
) -> str:
    if df.empty:
        body = "% No rows available."
        n_cols = 1
        align = "l"
    else:
        n_cols = df.shape[1]
        align = align or ("l" * n_cols)

        lines = []
        headers = [latex_escape(c) for c in df.columns]

        lines.append(r"\toprule")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")

        for _, row in df.iterrows():
            values = [latex_escape(row[col]) for col in df.columns]
            lines.append(" & ".join(values) + r" \\")

        lines.append(r"\bottomrule")
        body = "\n".join(lines)

    table_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        font_size,
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
    ]

    if resize:
        table_lines.append(r"\resizebox{\textwidth}{!}{%")

    table_lines.append(rf"\begin{{tabular}}{{{align}}}")
    table_lines.append(body)
    table_lines.append(r"\end{tabular}")

    if resize:
        table_lines.append(r"}")

    table_lines.append(r"\end{table}")
    table_lines.append("")

    return "\n".join(table_lines)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.read_csv(path)


def prepare_claim_summary_table(cross_dir: Path) -> pd.DataFrame:
    claim = read_csv_required(cross_dir / "cross_dataset_claim_summary_by_dataset.csv")

    out = pd.DataFrame()
    out["Dataset"] = claim["dataset_key"].map(dataset_label)
    out["Subset $<$ CMIM JMI"] = (
        claim["models_where_subset_beats_cmim_on_joint_mi"].astype(int).astype(str)
        + "/"
        + claim["n_models"].astype(int).astype(str)
    )
    out["Subset $<$ fairness baseline JMI"] = (
        claim["models_where_subset_beats_best_fairness_baseline_on_joint_mi"]
        .astype(int)
        .astype(str)
        + "/"
        + claim["n_models"].astype(int).astype(str)
    )
    out[r"Mean $\Delta$JMI vs CMIM"] = claim[
        "mean_subset_joint_mi_advantage_vs_cmim"
    ].map(lambda x: fmt_num(x, 4))
    out[r"Mean $\Delta$JMI vs baseline"] = claim[
        "mean_subset_joint_mi_advantage_vs_best_fairness_baseline"
    ].map(lambda x: fmt_num(x, 4))
    out["Interpretation"] = claim["interpretation"].map(
        lambda x: INTERPRETATION_LABELS.get(str(x), str(x))
    )

    return out


def prepare_subset_vs_baseline_table(cross_dir: Path) -> pd.DataFrame:
    svb = read_csv_required(cross_dir / "cross_dataset_subset_vs_baselines.csv")

    out = pd.DataFrame()
    out["Dataset"] = svb["dataset_key"].map(dataset_label)
    out["Model"] = svb["model"].map(model_label)
    out["Subset selector"] = svb["subset_selector"].map(compact_selector)
    out["Best fairness baseline"] = svb["best_fairness_baseline_selector"].map(
        compact_selector
    )
    out["Subset Acc."] = svb["subset_accuracy"].map(lambda x: fmt_num(x, 4))
    out["Baseline Acc."] = svb["baseline_accuracy"].map(lambda x: fmt_num(x, 4))
    out["Subset JMI"] = svb["subset_joint_mi"].map(lambda x: fmt_num(x, 4))
    out["Baseline JMI"] = svb["baseline_joint_mi"].map(lambda x: fmt_num(x, 4))
    out[r"$\Delta$JMI"] = svb[
        "subset_joint_mi_advantage_vs_best_fairness_baseline"
    ].map(lambda x: fmt_num(x, 4))
    out[r"$\Delta$Acc."] = svb[
        "subset_accuracy_delta_vs_best_fairness_baseline"
    ].map(lambda x: fmt_num(x, 4))

    return out.sort_values(["Dataset", "Model"])


def prepare_best_proxy_table(cross_dir: Path) -> pd.DataFrame:
    best = read_csv_required(cross_dir / "cross_dataset_best_by_dataset_model.csv")

    best = best[best["criterion"] == "lowest_joint_subset_mi"].copy()

    out = pd.DataFrame()
    out["Dataset"] = best["dataset_key"].map(dataset_label)
    out["Model"] = best["model"].map(model_label)
    out["Best selector by JMI"] = best["selector"].map(compact_selector)
    out["Family"] = best["selector_family"]
    out["Acc."] = best["accuracy"].map(lambda x: fmt_num(x, 4))
    out["JMI"] = best["joint_subset_mi_sensitive"].map(lambda x: fmt_num(x, 4))
    out["Attacker BA"] = best["sensitive_attacker_balanced_accuracy"].map(
        lambda x: fmt_num(x, 4)
    )
    out["DPD"] = best["dpd"].map(lambda x: fmt_num(x, 4))
    out["EOdds"] = best["equalized_odds_difference"].map(lambda x: fmt_num(x, 4))

    return out.sort_values(["Dataset", "Model"])


def prepare_family_compact_table(cross_dir: Path) -> pd.DataFrame:
    all_results = read_csv_required(cross_dir / "cross_dataset_all_results.csv")

    needed = [
        "dataset_key",
        "selector_family",
        "accuracy",
        "joint_subset_mi_sensitive",
        "sensitive_attacker_balanced_accuracy",
        "dpd",
        "equalized_odds_difference",
    ]

    missing = [c for c in needed if c not in all_results.columns]
    if missing:
        raise ValueError(f"Missing columns in cross_dataset_all_results.csv: {missing}")

    summary = (
        all_results.groupby(["dataset_key", "selector_family"], dropna=False)
        .agg(
            best_accuracy=("accuracy", "max"),
            lowest_joint_mi=("joint_subset_mi_sensitive", "min"),
            lowest_attacker_ba=("sensitive_attacker_balanced_accuracy", "min"),
            lowest_dpd=("dpd", "min"),
            lowest_eodds=("equalized_odds_difference", "min"),
        )
        .reset_index()
    )

    family_rank = {family: i for i, family in enumerate(FAMILY_ORDER)}
    summary["family_rank"] = summary["selector_family"].map(
        lambda x: family_rank.get(str(x), 999)
    )

    summary = summary.sort_values(["dataset_key", "family_rank"])

    out = pd.DataFrame()
    out["Dataset"] = summary["dataset_key"].map(dataset_label)
    out["Selector family"] = summary["selector_family"]
    out["Best Acc."] = summary["best_accuracy"].map(lambda x: fmt_num(x, 4))
    out["Lowest JMI"] = summary["lowest_joint_mi"].map(lambda x: fmt_num(x, 4))
    out["Lowest attacker BA"] = summary["lowest_attacker_ba"].map(lambda x: fmt_num(x, 4))
    out["Lowest DPD"] = summary["lowest_dpd"].map(lambda x: fmt_num(x, 4))
    out["Lowest EOdds"] = summary["lowest_eodds"].map(lambda x: fmt_num(x, 4))

    return out


def save_tables(cross_dir: Path, tables_dir: Path) -> dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    claim_table = prepare_claim_summary_table(cross_dir)
    subset_table = prepare_subset_vs_baseline_table(cross_dir)
    best_proxy_table = prepare_best_proxy_table(cross_dir)
    family_table = prepare_family_compact_table(cross_dir)

    table_specs = [
        (
            "table_cross_dataset_claim_summary.tex",
            claim_table,
            "Cross-dataset support for subset-aware proxy-leakage control.",
            "tab:cross-dataset-claim-summary",
        ),
        (
            "table_subset_vs_fairness_baselines.tex",
            subset_table,
            "Subset-aware FA-CMIM compared with the best fairness-aware baseline by dataset and model.",
            "tab:subset-vs-fairness-baselines",
        ),
        (
            "table_best_proxy_leakage_by_dataset_model.tex",
            best_proxy_table,
            "Best selector by joint subset mutual information for each dataset and downstream model.",
            "tab:best-proxy-leakage",
        ),
        (
            "table_selector_family_compact_summary.tex",
            family_table,
            "Compact selector-family summary across datasets.",
            "tab:selector-family-summary",
        ),
    ]

    for filename, table_df, caption, label in table_specs:
        path = tables_dir / filename
        latex = make_latex_table(
            table_df,
            caption=caption,
            label=label,
            resize=True,
        )
        path.write_text(latex, encoding="utf-8")
        outputs[filename] = path

    return outputs


def figure_caption_from_stem(stem: str) -> str:
    readable = stem.replace("_", " ").replace("-", " ")

    if "accuracy_vs_joint_mi" in stem:
        return "Accuracy versus joint subset mutual information across datasets and models."

    if "subset_vs_fairness_baseline" in stem:
        return "Joint-MI advantage of Subset-aware FA-CMIM over the best fairness-aware baseline."

    if "dpd" in stem:
        return "Pareto frontier for accuracy and demographic parity difference."

    if "equal_opportunity" in stem:
        return "Pareto frontier for accuracy and equal opportunity difference."

    if "equalized_odds" in stem:
        return "Pareto frontier for accuracy and equalized odds difference."

    if "joint_subset_mi_sensitive" in stem:
        return "Pareto frontier for accuracy and joint sensitive leakage."

    return f"Generated result figure: {readable}."


def collect_figures(figure_roots: list[Path], figures_dir: Path) -> list[dict[str, str]]:
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for root in figure_roots:
        if not root.exists():
            continue

        dataset_hint = "figure"

        parts = [p.lower() for p in root.parts]

        if "adult" in parts:
            dataset_hint = "adult"
        elif "acs_income" in parts:
            dataset_hint = "acs-income"
        elif "compas" in parts:
            dataset_hint = "compas"
        elif "german_credit" in parts:
            dataset_hint = "german-credit"
        elif "cross_dataset" in parts:
            dataset_hint = "cross-dataset"

        for src in sorted(root.glob("*.png")):
            dst_name = f"{dataset_hint}_{safe_label(src.stem)}.png"
            dst = figures_dir / dst_name
            shutil.copy2(src, dst)

            records.append(
                {
                    "source": str(src),
                    "filename": dst_name,
                    "relative_path": f"figures/{dst_name}",
                    "caption": figure_caption_from_stem(src.stem),
                    "label": f"fig:{safe_label(dst_name.replace('.png', ''))}",
                }
            )

    return records


def make_figure_references(records: list[dict[str, str]]) -> str:
    lines = [
        "% Auto-generated figure references.",
        "% Requires: \\usepackage{graphicx}",
        "",
    ]

    if not records:
        lines.append("% No figure files found.")
        return "\n".join(lines)

    for record in records:
        lines.extend(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.92\textwidth]{{{record['relative_path']}}}",
                rf"\caption{{{latex_escape(record['caption'])}}}",
                rf"\label{{{record['label']}}}",
                r"\end{figure}",
                "",
            ]
        )

    return "\n".join(lines)


def save_figure_references(
    *,
    figure_roots: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    records = collect_figures(figure_roots, figures_dir)

    refs_path = output_dir / "figure_references.tex"
    refs_path.write_text(make_figure_references(records), encoding="utf-8")

    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return {
        "figure_references": refs_path,
        "figure_manifest": manifest_path,
    }


def save_index(output_dir: Path, table_paths: dict[str, Path], figure_paths: dict[str, Path]) -> Path:
    lines = [
        "# LaTeX artifact index",
        "",
        "## Required LaTeX packages",
        "",
        "```latex",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{adjustbox}",
        "```",
        "",
        "## Tables",
        "",
    ]

    for name, path in sorted(table_paths.items()):
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- `{figure_paths['figure_references']}`",
            f"- `{figure_paths['figure_manifest']}`",
            "",
            "## Suggested manuscript inclusion order",
            "",
            "```latex",
            r"\input{tables/table_cross_dataset_claim_summary}",
            r"\input{tables/table_subset_vs_fairness_baselines}",
            r"\input{tables/table_best_proxy_leakage_by_dataset_model}",
            r"\input{tables/table_selector_family_compact_summary}",
            r"\input{figure_references}",
            "```",
            "",
        ]
    )

    index_path = output_dir / "latex_artifacts_index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")

    return index_path


def run_latex_artifact_generation(args) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"

    cross_dir = Path(args.cross_dataset_dir)

    if args.figure_roots:
        figure_roots = [Path(p) for p in args.figure_roots]
    else:
        figure_roots = [
            Path("results/cross_dataset/plots"),
            Path("results/adult/final/plots"),
            Path("results/acs_income/final/plots"),
            Path("results/compas/final/plots"),
            Path("results/german_credit/final/plots"),
        ]

    output_dir.mkdir(parents=True, exist_ok=True)

    table_paths = save_tables(cross_dir, tables_dir)
    figure_paths = save_figure_references(
        figure_roots=figure_roots,
        output_dir=output_dir,
    )
    index_path = save_index(output_dir, table_paths, figure_paths)

    outputs = {
        **table_paths,
        **figure_paths,
        "index": index_path,
    }

    print("\nGenerated LaTeX artifacts:")
    for name, path in outputs.items():
        print(f"{name}: {path}")

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-ready LaTeX tables and figure references."
    )

    parser.add_argument(
        "--cross_dataset_dir",
        default="results/cross_dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="manuscript/latex",
    )
    parser.add_argument(
        "--figure_roots",
        nargs="*",
        default=None,
        help="Optional list of directories containing PNG figures.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_latex_artifact_generation(args)


if __name__ == "__main__":
    main()
