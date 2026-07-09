
from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_selected_features(value):
    """
    Robust parser for selected feature strings.

    Handles:
    - pipe-separated: age|education|occupation
    - Python list string: ['age', 'education']
    - comma-separated: age, education
    - semicolon-separated
    """
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, list):
        return [str(v).strip().strip("'\"") for v in value if str(v).strip()]

    text_value = str(value).strip()

    if not text_value or text_value.lower() in {"nan", "none", "[]"}:
        return []

    # Try Python list first.
    try:
        parsed = ast.literal_eval(text_value)
        if isinstance(parsed, list):
            return [
                str(v).strip().strip("'\"")
                for v in parsed
                if str(v).strip()
            ]
    except Exception:
        pass

    text_value = text_value.strip("[]")

    if "|" in text_value:
        parts = text_value.split("|")
    elif ";" in text_value:
        parts = text_value.split(";")
    elif "," in text_value:
        parts = text_value.split(",")
    else:
        parts = [text_value]

    cleaned = []
    for p in parts:
        f = str(p).strip().strip("'\"")
        if f and f.lower() not in {"nan", "none"}:
            cleaned.append(f)

    return cleaned


def compute_selection_frequencies(results_path: Path):
    results = pd.read_csv(results_path)

    feature_col_candidates = [
        c for c in results.columns
        if "selected" in c.lower() and "feature" in c.lower()
    ]

    if not feature_col_candidates:
        raise ValueError(f"No selected-feature column found in {results_path}")

    feature_col = feature_col_candidates[0]

    # Avoid triple-counting the same selector because results are repeated
    # for LR, RF, and GB models.
    dedup_cols = [
        c for c in [
            "dataset",
            "selector",
            "k",
            "lambda",
            "lambda_values",
            feature_col,
        ]
        if c in results.columns
    ]

    unique_selectors = results[dedup_cols].drop_duplicates()

    counter = Counter()
    denominator = 0

    for value in unique_selectors[feature_col]:
        features = set(parse_selected_features(value))
        if not features:
            continue

        denominator += 1
        counter.update(features)

    counts = dict(counter)
    frequencies = {
        feature: count / denominator
        for feature, count in counts.items()
    } if denominator > 0 else {}

    return counts, frequencies, denominator


def update_diagnostic_table(
    diagnostic_path: Path,
    results_path: Path,
    top_n: int = 15,
):
    diagnostic = pd.read_csv(diagnostic_path)
    counts, frequencies, denominator = compute_selection_frequencies(results_path)

    diagnostic["selection_count"] = (
        diagnostic["feature"].map(counts).fillna(0).astype(int)
    )

    diagnostic["selection_frequency"] = (
        diagnostic["feature"].map(frequencies).fillna(0.0).astype(float)
    )

    diagnostic["selection_denominator"] = denominator

    # Optional manuscript-friendly score: high only when a feature is both
    # entangled and frequently selected.
    diagnostic["entanglement_selection_score"] = (
        diagnostic["entanglement_score"] * diagnostic["selection_frequency"]
    )

    diagnostic = diagnostic.sort_values(
        ["entanglement_score", "selection_frequency"],
        ascending=[False, False],
    )

    diagnostic.to_csv(diagnostic_path, index=False)

    dataset_name = diagnostic_path.name.replace("_entangled_proxy_diagnostics.csv", "")
    output_dir = diagnostic_path.parent

    top_path = output_dir / f"{dataset_name}_top_entangled_proxies.csv"
    top_tex_path = output_dir / f"{dataset_name}_top_entangled_proxies.tex"

    top = diagnostic.head(top_n)
    top.to_csv(top_path, index=False)
    top.to_latex(
        top_tex_path,
        index=False,
        float_format="%.4f",
        escape=True,
    )

    selected_top_path = output_dir / f"{dataset_name}_top_entangled_selected_proxies.csv"
    selected_top_tex_path = output_dir / f"{dataset_name}_top_entangled_selected_proxies.tex"

    selected_top = diagnostic.sort_values(
        ["entanglement_selection_score", "entanglement_score"],
        ascending=[False, False],
    ).head(top_n)

    selected_top.to_csv(selected_top_path, index=False)
    selected_top.to_latex(
        selected_top_tex_path,
        index=False,
        float_format="%.4f",
        escape=True,
    )

    print("\nUpdated:", diagnostic_path)
    print("Selection denominator:", denominator)
    print("Saved:", top_path)
    print("Saved:", selected_top_path)
    print()
    print(top[[
        "feature",
        "mi_feature_sensitive",
        "mi_feature_target",
        "cmi_feature_target_given_sensitive",
        "entanglement_score",
        "selection_count",
        "selection_frequency",
        "entanglement_selection_score",
    ]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["adult", "german_credit"])
    parser.add_argument("--top_n", type=int, default=15)
    args = parser.parse_args()

    if args.dataset == "adult":
        results_path = Path("results/adult/final/adult_pilot_results.csv")
        diagnostic_paths = sorted(
            Path("results/proxy_entanglement").rglob("adult_entangled_proxy_diagnostics.csv")
        )
    else:
        results_path = Path("results/german_credit/final/german_credit_pilot_results.csv")
        diagnostic_paths = sorted(
            Path("results/proxy_entanglement").rglob("german_credit_entangled_proxy_diagnostics.csv")
        )

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    if not diagnostic_paths:
        raise FileNotFoundError(f"No diagnostic files found for {args.dataset}")

    print("Using results file:", results_path)
    print("Diagnostic files to update:")
    for p in diagnostic_paths:
        print(" ", p)

    for diagnostic_path in diagnostic_paths:
        update_diagnostic_table(
            diagnostic_path=diagnostic_path,
            results_path=results_path,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
