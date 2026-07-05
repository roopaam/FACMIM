import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.summary import (
    compute_pareto_frontier,
    plot_all_pareto,
    save_summary_outputs,
)


def make_results():
    return pd.DataFrame(
        [
            {
                "selector": "A_high_accuracy",
                "model": "logistic_regression",
                "accuracy": 0.90,
                "balanced_accuracy": 0.88,
                "f1": 0.89,
                "auroc": 0.91,
                "dpd": 0.20,
                "equal_opportunity_difference": 0.18,
                "equalized_odds_difference": 0.22,
                "joint_subset_mi_sensitive": 0.30,
                "selected_feature_count": 5,
                "selected_features": "x1|x2|x3",
            },
            {
                "selector": "B_balanced",
                "model": "logistic_regression",
                "accuracy": 0.88,
                "balanced_accuracy": 0.86,
                "f1": 0.87,
                "auroc": 0.90,
                "dpd": 0.05,
                "equal_opportunity_difference": 0.06,
                "equalized_odds_difference": 0.08,
                "joint_subset_mi_sensitive": 0.10,
                "selected_feature_count": 5,
                "selected_features": "x1|x4|x5",
            },
            {
                "selector": "C_dominated",
                "model": "logistic_regression",
                "accuracy": 0.85,
                "balanced_accuracy": 0.83,
                "f1": 0.84,
                "auroc": 0.86,
                "dpd": 0.10,
                "equal_opportunity_difference": 0.12,
                "equalized_odds_difference": 0.13,
                "joint_subset_mi_sensitive": 0.15,
                "selected_feature_count": 5,
                "selected_features": "x2|x4|x6",
            },
            {
                "selector": "D_rf",
                "model": "random_forest",
                "accuracy": 0.91,
                "balanced_accuracy": 0.89,
                "f1": 0.90,
                "auroc": 0.92,
                "dpd": 0.12,
                "equal_opportunity_difference": 0.10,
                "equalized_odds_difference": 0.11,
                "joint_subset_mi_sensitive": 0.20,
                "selected_feature_count": 5,
                "selected_features": "x1|x7|x8",
            },
        ]
    )


def test_compute_pareto_frontier_marks_expected_rows():
    df = make_results()

    out = compute_pareto_frontier(
        df,
        utility_col="accuracy",
        fairness_col="dpd",
        group_cols=("model",),
    )

    lr = out[out["model"] == "logistic_regression"]

    pareto_selectors = set(lr.loc[lr["pareto_optimal"], "selector"])

    assert "A_high_accuracy" in pareto_selectors
    assert "B_balanced" in pareto_selectors
    assert "C_dominated" not in pareto_selectors


def test_save_summary_outputs_creates_expected_files(tmp_path):
    df = make_results()
    results_path = tmp_path / "adult_pilot_results.csv"
    output_dir = tmp_path / "summary"

    df.to_csv(results_path, index=False)

    paths = save_summary_outputs(
        results_path=results_path,
        output_dir=output_dir,
        utility_col="accuracy",
        fairness_col="dpd",
    )

    for path in paths.values():
        assert Path(path).exists()

    summary = pd.read_csv(paths["selector_model_summary"])
    pareto = pd.read_csv(paths["pareto"])

    assert not summary.empty
    assert not pareto.empty
    assert "pareto_optimal" in pareto.columns
    assert "tradeoff_score" in pd.read_csv(paths["scored_results"]).columns


def test_plot_all_pareto_creates_png_files(tmp_path):
    df = make_results()
    output_dir = tmp_path / "plots"

    paths = plot_all_pareto(
        df,
        output_dir=output_dir,
        utility_col="accuracy",
        fairness_cols=["dpd"],
        group_col="model",
        label_col="selector",
    )

    assert len(paths) >= 1

    for path in paths:
        p = Path(path)
        assert p.exists()
        assert p.suffix == ".png"
        assert p.stat().st_size > 0


def test_summary_and_plot_cli_smoke(tmp_path):
    df = make_results()
    results_path = tmp_path / "adult_pilot_results.csv"
    summary_dir = tmp_path / "summary"
    plots_dir = tmp_path / "plots"

    df.to_csv(results_path, index=False)

    summary_cmd = [
        sys.executable,
        "scripts/summarize_results.py",
        "--results_path",
        str(results_path),
        "--output_dir",
        str(summary_dir),
    ]

    plot_cmd = [
        sys.executable,
        "scripts/plot_pareto.py",
        "--results_path",
        str(results_path),
        "--output_dir",
        str(plots_dir),
        "--fairness_cols",
        "dpd",
    ]

    summary_result = subprocess.run(
        summary_cmd,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert summary_result.returncode == 0, summary_result.stderr
    assert (summary_dir / "selector_model_summary.csv").exists()

    plot_result = subprocess.run(
        plot_cmd,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert plot_result.returncode == 0, plot_result.stderr
    assert list(plots_dir.glob("*.png"))
