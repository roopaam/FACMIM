
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.proxy_entanglement import (
    compute_entangled_proxy_table,
    evaluate_subset,
    load_fairness_dataset,
    select_all_ablation_methods,
)


def run_variant(
    dataset_name,
    X,
    y,
    s,
    variant_name,
    removed_features,
    k,
    lambdas,
    models,
    random_state,
    test_size,
    n_bins,
):
    X_variant = X.drop(columns=[f for f in removed_features if f in X.columns]).copy()

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_variant,
        y,
        s,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    selectors = select_all_ablation_methods(
        X=X_train,
        y=y_train,
        s=s_train,
        k=k,
        lambdas=lambdas,
        n_bins=n_bins,
        random_state=random_state,
    )

    rows = []

    for selector_name, lam, selected_features in selectors:
        for model_name in models:
            metrics = evaluate_subset(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                s_train=s_train,
                s_test=s_test,
                selected_features=selected_features,
                model_name=model_name,
                random_state=random_state,
                n_bins=n_bins,
            )

            rows.append({
                "dataset": dataset_name,
                "variant": variant_name,
                "removed_features": removed_features,
                "selector": selector_name,
                "lambda": lam,
                "model": model_name,
                "k": k,
                "selected_features": selected_features,
                **metrics,
            })

    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results
        .groupby(["dataset", "variant", "selector"], dropna=False)
        .agg(
            best_accuracy=("accuracy", "max"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_joint_mi=("joint_subset_mi_sensitive", "mean"),
            min_joint_mi=("joint_subset_mi_sensitive", "min"),
            mean_attacker_ba=("sensitive_attacker_balanced_accuracy", "mean"),
            min_attacker_ba=("sensitive_attacker_balanced_accuracy", "min"),
            mean_dpd=("dpd", "mean"),
            mean_eod=("equal_opportunity_difference", "mean"),
            mean_eodds=("equalized_odds_difference", "mean"),
        )
        .reset_index()
    )
    return summary


def create_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    original = summary[summary["variant"] == "original"].copy()
    removed = summary[summary["variant"] == "remove_entangled_proxies"].copy()

    merged = original.merge(
        removed,
        on=["dataset", "selector"],
        suffixes=("_original", "_removed"),
    )

    merged["joint_mi_drop_after_proxy_removal"] = (
        merged["mean_joint_mi_original"] - merged["mean_joint_mi_removed"]
    )

    merged["attacker_ba_drop_after_proxy_removal"] = (
        merged["mean_attacker_ba_original"] - merged["mean_attacker_ba_removed"]
    )

    merged["accuracy_change_after_proxy_removal"] = (
        merged["mean_accuracy_removed"] - merged["mean_accuracy_original"]
    )

    keep_cols = [
        "dataset",
        "selector",
        "mean_accuracy_original",
        "mean_accuracy_removed",
        "accuracy_change_after_proxy_removal",
        "mean_joint_mi_original",
        "mean_joint_mi_removed",
        "joint_mi_drop_after_proxy_removal",
        "mean_attacker_ba_original",
        "mean_attacker_ba_removed",
        "attacker_ba_drop_after_proxy_removal",
    ]

    return merged[keep_cols].sort_values(
        ["joint_mi_drop_after_proxy_removal", "mean_joint_mi_removed"],
        ascending=[False, True],
    )


def create_best_leakage_table(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary
        .sort_values(["dataset", "variant", "mean_joint_mi", "mean_attacker_ba"])
        .groupby(["dataset", "variant"], as_index=False)
        .head(10)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["adult", "german_credit"])
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--top_n_remove", type=int, default=2)
    parser.add_argument("--drop_features", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "gradient_boosting"],
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    X, y, s, meta = load_fairness_dataset(
        dataset=args.dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
        data_path=args.data_path,
    )

    diag = compute_entangled_proxy_table(
        X=X,
        y=y,
        s=s,
        results_path=args.results_path,
        n_bins=args.n_bins,
    )

    if args.drop_features:
        removed_features = [f for f in args.drop_features if f in X.columns]
    else:
        removed_features = diag.head(args.top_n_remove)["feature"].tolist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diag_path = output_dir / f"{args.dataset}_entangled_proxy_diagnostics.csv"
    diag.to_csv(diag_path, index=False)

    rows = []

    rows.extend(run_variant(
        dataset_name=args.dataset,
        X=X,
        y=y,
        s=s,
        variant_name="original",
        removed_features=[],
        k=args.k,
        lambdas=args.lambdas,
        models=args.models,
        random_state=args.random_state,
        test_size=args.test_size,
        n_bins=args.n_bins,
    ))

    rows.extend(run_variant(
        dataset_name=args.dataset,
        X=X,
        y=y,
        s=s,
        variant_name="remove_entangled_proxies",
        removed_features=removed_features,
        k=args.k,
        lambdas=args.lambdas,
        models=args.models,
        random_state=args.random_state,
        test_size=args.test_size,
        n_bins=args.n_bins,
    ))

    results = pd.DataFrame(rows)
    summary = summarize(results)
    delta = create_delta_table(summary)
    best_leakage = create_best_leakage_table(summary)

    results_path = output_dir / f"{args.dataset}_all_methods_proxy_removal_ablation_results.csv"
    summary_path = output_dir / f"{args.dataset}_all_methods_proxy_removal_ablation_summary.csv"
    delta_path = output_dir / f"{args.dataset}_all_methods_proxy_removal_ablation_delta.csv"
    best_path = output_dir / f"{args.dataset}_all_methods_best_leakage_by_variant.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    delta.to_csv(delta_path, index=False)
    best_leakage.to_csv(best_path, index=False)

    latex_summary_path = output_dir / f"{args.dataset}_all_methods_proxy_removal_ablation_summary.tex"
    latex_best_path = output_dir / f"{args.dataset}_all_methods_best_leakage_by_variant.tex"

    summary.sort_values(["variant", "mean_joint_mi"]).head(40).to_latex(
        latex_summary_path,
        index=False,
        float_format="%.4f",
    )

    best_leakage.to_latex(
        latex_best_path,
        index=False,
        float_format="%.4f",
    )

    print("Metadata:", meta)
    print("Removed entangled proxies:", removed_features)
    print("Saved diagnostics:", diag_path)
    print("Saved full results:", results_path)
    print("Saved summary:", summary_path)
    print("Saved delta:", delta_path)
    print("Saved best leakage table:", best_path)
    print()
    print("Top entangled proxies:")
    print(diag.head(10).to_string(index=False))
    print()
    print("Best leakage by variant:")
    print(best_leakage.to_string(index=False))
    print()
    print("Largest leakage drops after proxy removal:")
    print(delta.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
