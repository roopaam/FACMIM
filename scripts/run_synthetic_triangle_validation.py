
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.proxy_entanglement import (
    compute_entangled_proxy_table,
    evaluate_subset,
    select_cmim,
    select_subset_fa_cmim,
)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def flip_bits(x, noise, rng):
    flips = rng.binomial(1, noise, size=len(x))
    return np.logical_xor(x.astype(bool), flips.astype(bool)).astype(int)


def generate_synthetic_triangle(n: int, scenario: str, seed: int):
    rng = np.random.default_rng(seed)

    S = rng.binomial(1, 0.5, size=n)
    Z = rng.binomial(1, 0.5, size=n)

    if scenario == "triangle_entangled":
        # S -> P, S -> Y, P -> Y
        P1 = flip_bits(S, 0.15, rng)
        P2 = flip_bits(S, 0.25, rng)
        T1 = flip_bits(Z, 0.10, rng)
        T2 = flip_bits(Z, 0.20, rng)

        prob_y = sigmoid(-1.0 + 1.2 * Z + 1.0 * P1 + 0.6 * P2 + 0.4 * S)
        Y = rng.binomial(1, prob_y)

        X = pd.DataFrame({
            "P1_proxy": P1,
            "P2_proxy": P2,
            "T1_target": T1,
            "T2_target": T2,
        })

    elif scenario == "target_only_no_proxy":
        # Target-relevant features exist, but they do not encode S.
        P1 = rng.binomial(1, 0.5, size=n)
        P2 = rng.binomial(1, 0.5, size=n)
        T1 = flip_bits(Z, 0.10, rng)
        T2 = flip_bits(Z, 0.20, rng)

        prob_y = sigmoid(-0.8 + 1.8 * T1 + 1.2 * T2)
        Y = rng.binomial(1, prob_y)

        X = pd.DataFrame({
            "P1_irrelevant": P1,
            "P2_irrelevant": P2,
            "T1_target": T1,
            "T2_target": T2,
        })

    elif scenario == "sensitive_only_not_target":
        # Sensitive proxies exist, but they are not target-relevant.
        P1 = flip_bits(S, 0.10, rng)
        P2 = flip_bits(S, 0.20, rng)
        T1 = flip_bits(Z, 0.10, rng)
        T2 = flip_bits(Z, 0.20, rng)

        prob_y = sigmoid(-0.8 + 1.8 * T1 + 1.2 * T2)
        Y = rng.binomial(1, prob_y)

        X = pd.DataFrame({
            "P1_sensitive_only": P1,
            "P2_sensitive_only": P2,
            "T1_target": T1,
            "T2_target": T2,
        })

    elif scenario == "weak_joint_entangled":
        # Multiple weak proxies jointly encode S and contribute to Y.
        P1 = flip_bits(S, 0.30, rng)
        P2 = flip_bits(S, 0.30, rng)
        P3 = flip_bits(S, 0.30, rng)
        T1 = flip_bits(Z, 0.10, rng)
        T2 = flip_bits(Z, 0.25, rng)

        proxy_sum = P1 + P2 + P3
        prob_y = sigmoid(-1.2 + 1.0 * Z + 0.55 * proxy_sum + 0.3 * S)
        Y = rng.binomial(1, prob_y)

        X = pd.DataFrame({
            "P1_weak_proxy": P1,
            "P2_weak_proxy": P2,
            "P3_weak_proxy": P3,
            "T1_target": T1,
            "T2_target": T2,
        })

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    for i in range(1, 7):
        X[f"N{i}_noise"] = rng.binomial(1, 0.5, size=n)

    return X, Y.astype(int), S.astype(int)


def run_one(seed, scenario, n, k, lambdas, models, test_size, n_bins):
    X, y, s = generate_synthetic_triangle(n=n, scenario=scenario, seed=seed)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        s,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    rows = []

    cmim_features = select_cmim(X_train, y_train, k=k, n_bins=n_bins)
    selectors = [("CMIM", None, cmim_features)]

    for lam in lambdas:
        subset_features = select_subset_fa_cmim(
            X_train,
            y_train,
            s_train,
            k=k,
            lambda_value=lam,
            n_bins=n_bins,
        )
        selectors.append((f"SubsetFACMIM_lambda{lam}", lam, subset_features))

    for selector_name, lam, features in selectors:
        for model_name in models:
            metrics = evaluate_subset(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                s_train=s_train,
                s_test=s_test,
                selected_features=features,
                model_name=model_name,
                random_state=seed,
                n_bins=n_bins,
            )

            rows.append({
                "seed": seed,
                "scenario": scenario,
                "selector": selector_name,
                "lambda": lam,
                "model": model_name,
                "k": k,
                "selected_features": features,
                **metrics,
            })

    diag = compute_entangled_proxy_table(X, y, s, results_path=None, n_bins=n_bins)
    diag["seed"] = seed
    diag["scenario"] = scenario

    return rows, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "triangle_entangled",
            "target_only_no_proxy",
            "sensitive_only_not_target",
            "weak_joint_entangled",
        ],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "gradient_boosting"],
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_diag = []

    for scenario in args.scenarios:
        print(f"Running scenario: {scenario}")
        for seed in args.seeds:
            print(f"  seed={seed}")
            rows, diag = run_one(
                seed=seed,
                scenario=scenario,
                n=args.n,
                k=args.k,
                lambdas=args.lambdas,
                models=args.models,
                test_size=args.test_size,
                n_bins=args.n_bins,
            )
            all_rows.extend(rows)
            all_diag.append(diag)

    results = pd.DataFrame(all_rows)
    diagnostics = pd.concat(all_diag, ignore_index=True)

    results_path = output_dir / "synthetic_triangle_validation_results.csv"
    diag_path = output_dir / "synthetic_triangle_entanglement_diagnostics.csv"

    results.to_csv(results_path, index=False)
    diagnostics.to_csv(diag_path, index=False)

    summary = (
        results
        .groupby(["scenario", "selector"], dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_joint_mi=("joint_subset_mi_sensitive", "mean"),
            std_joint_mi=("joint_subset_mi_sensitive", "std"),
            mean_attacker_ba=("sensitive_attacker_balanced_accuracy", "mean"),
            std_attacker_ba=("sensitive_attacker_balanced_accuracy", "std"),
            mean_dpd=("dpd", "mean"),
            mean_eod=("equal_opportunity_difference", "mean"),
            mean_eodds=("equalized_odds_difference", "mean"),
        )
        .reset_index()
    )

    summary_path = output_dir / "synthetic_triangle_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    latex_path = output_dir / "synthetic_triangle_validation_summary.tex"
    summary.to_latex(latex_path, index=False, float_format="%.4f")

    print("\nSaved results:", results_path)
    print("Saved diagnostics:", diag_path)
    print("Saved summary:", summary_path)
    print("Saved LaTeX summary:", latex_path)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
