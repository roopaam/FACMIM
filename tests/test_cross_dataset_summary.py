import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def make_fake_results(path: Path, dataset: str, adult_like: bool):
    rows = []

    base_rows = [
        {
            "selector": "CMIM_k8",
            "model": "gradient_boosting",
            "accuracy": 0.80,
            "balanced_accuracy": 0.78,
            "f1": 0.70,
            "auroc": 0.90,
            "dpd": 0.20,
            "dpr": 0.50,
            "equal_opportunity_difference": 0.18,
            "equalized_odds_difference": 0.22,
            "joint_subset_mi_sensitive": 0.55,
            "sensitive_attacker_balanced_accuracy": 0.80,
            "selected_features": "a|b|c",
        },
        {
            "selector": "FairCFS_k8_lambda1.0",
            "model": "gradient_boosting",
            "accuracy": 0.84,
            "balanced_accuracy": 0.80,
            "f1": 0.72,
            "auroc": 0.91,
            "dpd": 0.18,
            "dpr": 0.55,
            "equal_opportunity_difference": 0.16,
            "equalized_odds_difference": 0.20,
            "joint_subset_mi_sensitive": 0.58,
            "sensitive_attacker_balanced_accuracy": 0.82,
            "selected_features": "a|b|d",
        },
        {
            "selector": "SubsetFACMIM_k8_lambda1.0",
            "model": "gradient_boosting",
            "accuracy": 0.82,
            "balanced_accuracy": 0.79,
            "f1": 0.71,
            "auroc": 0.90,
            "dpd": 0.10,
            "dpr": 0.70,
            "equal_opportunity_difference": 0.10,
            "equalized_odds_difference": 0.12,
            "joint_subset_mi_sensitive": 0.25 if adult_like else 0.58,
            "sensitive_attacker_balanced_accuracy": 0.62 if adult_like else 0.82,
            "selected_features": "a|e|f",
        },
    ]

    for row in base_rows:
        row = row.copy()
        row["dataset"] = dataset
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_cross_dataset_summary_script_runs(tmp_path):
    adult_path = tmp_path / "adult.csv"
    acs_path = tmp_path / "acs.csv"
    compas_path = tmp_path / "compas.csv"
    german_path = tmp_path / "german.csv"

    make_fake_results(adult_path, "Adult", adult_like=True)
    make_fake_results(acs_path, "ACSIncome", adult_like=False)
    make_fake_results(compas_path, "COMPAS", adult_like=False)
    make_fake_results(german_path, "German Credit", adult_like=True)

    output_dir = tmp_path / "cross_dataset"

    cmd = [
        sys.executable,
        "scripts/create_cross_dataset_summary.py",
        "--adult_path",
        str(adult_path),
        "--acs_income_path",
        str(acs_path),
        "--compas_path",
        str(compas_path),
        "--german_credit_path",
        str(german_path),
        "--output_dir",
        str(output_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    expected_files = [
        "cross_dataset_all_results.csv",
        "cross_dataset_selector_family_summary.csv",
        "cross_dataset_best_by_dataset_model.csv",
        "cross_dataset_pareto_accuracy_joint_mi.csv",
        "cross_dataset_subset_vs_baselines.csv",
        "cross_dataset_claim_summary_by_dataset.csv",
        "cross_dataset_interpretation_notes.md",
    ]

    for filename in expected_files:
        assert (output_dir / filename).exists(), filename

    assert (output_dir / "plots" / "cross_dataset_accuracy_vs_joint_mi.png").exists()
    assert (
        output_dir / "plots" / "subset_vs_fairness_baseline_joint_mi_advantage.png"
    ).exists()


def test_cross_dataset_summary_detects_subset_advantage(tmp_path):
    adult_path = tmp_path / "adult.csv"
    acs_path = tmp_path / "acs.csv"

    make_fake_results(adult_path, "Adult", adult_like=True)
    make_fake_results(acs_path, "ACSIncome", adult_like=False)

    output_dir = tmp_path / "cross_dataset"

    cmd = [
        sys.executable,
        "scripts/create_cross_dataset_summary.py",
        "--adult_path",
        str(adult_path),
        "--acs_income_path",
        str(acs_path),
        "--compas_path",
        str(tmp_path / "missing_compas.csv"),
        "--german_credit_path",
        str(tmp_path / "missing_german.csv"),
        "--output_dir",
        str(output_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    claim = pd.read_csv(output_dir / "cross_dataset_claim_summary_by_dataset.csv")

    adult = claim[claim["dataset_key"] == "adult"].iloc[0]
    acs = claim[claim["dataset_key"] == "acs_income"].iloc[0]

    assert adult["models_where_subset_beats_best_fairness_baseline_on_joint_mi"] == 1
    assert acs["models_where_subset_beats_best_fairness_baseline_on_joint_mi"] == 0
