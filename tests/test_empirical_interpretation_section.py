import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_fake_cross_dataset_outputs(base: Path):
    base.mkdir(parents=True, exist_ok=True)

    claim = pd.DataFrame(
        [
            {
                "dataset_key": "adult",
                "n_models": 3,
                "models_where_subset_beats_cmim_on_joint_mi": 3,
                "models_where_subset_beats_best_fairness_baseline_on_joint_mi": 3,
                "mean_subset_joint_mi_advantage_vs_cmim": 0.20,
                "mean_subset_joint_mi_advantage_vs_best_fairness_baseline": 0.18,
                "interpretation": "strong_support_for_subset_proxy_leakage_advantage",
            },
            {
                "dataset_key": "acs_income",
                "n_models": 3,
                "models_where_subset_beats_cmim_on_joint_mi": 0,
                "models_where_subset_beats_best_fairness_baseline_on_joint_mi": 0,
                "mean_subset_joint_mi_advantage_vs_cmim": 0.00,
                "mean_subset_joint_mi_advantage_vs_best_fairness_baseline": 0.00,
                "interpretation": "flat_or_no_subset_proxy_leakage_advantage",
            },
        ]
    )

    svb = pd.DataFrame(
        [
            {
                "dataset_key": "adult",
                "model": "gradient_boosting",
                "subset_selector": "SubsetFACMIM_k8_lambda1.0",
                "subset_accuracy": 0.83,
                "subset_joint_mi": 0.25,
                "subset_attacker_ba": 0.63,
                "cmim_selector": "CMIM_k8",
                "cmim_accuracy": 0.84,
                "cmim_joint_mi": 0.59,
                "cmim_attacker_ba": 0.86,
                "subset_joint_mi_advantage_vs_cmim": 0.34,
                "subset_accuracy_delta_vs_cmim": -0.01,
                "best_fairness_baseline_selector": "FairCFS_k8_lambda1.0",
                "best_fairness_baseline_family": "FairCFS-style",
                "baseline_accuracy": 0.86,
                "baseline_joint_mi": 0.50,
                "baseline_attacker_ba": 0.79,
                "subset_joint_mi_advantage_vs_best_fairness_baseline": 0.25,
                "subset_accuracy_delta_vs_best_fairness_baseline": -0.03,
                "supports_subset_proxy_advantage_vs_cmim": True,
                "supports_subset_proxy_advantage_vs_fairness_baseline": True,
            },
            {
                "dataset_key": "acs_income",
                "model": "gradient_boosting",
                "subset_selector": "SubsetFACMIM_k8_lambda1.0",
                "subset_accuracy": 0.79,
                "subset_joint_mi": 0.66,
                "subset_attacker_ba": 0.59,
                "cmim_selector": "CMIM_k8",
                "cmim_accuracy": 0.79,
                "cmim_joint_mi": 0.66,
                "cmim_attacker_ba": 0.59,
                "subset_joint_mi_advantage_vs_cmim": 0.00,
                "subset_accuracy_delta_vs_cmim": 0.00,
                "best_fairness_baseline_selector": "FairCFS_k8_lambda1.0",
                "best_fairness_baseline_family": "FairCFS-style",
                "baseline_accuracy": 0.79,
                "baseline_joint_mi": 0.66,
                "baseline_attacker_ba": 0.59,
                "subset_joint_mi_advantage_vs_best_fairness_baseline": 0.00,
                "subset_accuracy_delta_vs_best_fairness_baseline": 0.00,
                "supports_subset_proxy_advantage_vs_cmim": False,
                "supports_subset_proxy_advantage_vs_fairness_baseline": False,
            },
        ]
    )

    dummy = pd.DataFrame(
        [
            {
                "dataset_key": "adult",
                "model": "gradient_boosting",
                "selector": "SubsetFACMIM_k8_lambda1.0",
                "selector_family": "SubsetFA-CMIM",
                "accuracy": 0.83,
                "joint_subset_mi_sensitive": 0.25,
                "sensitive_attacker_balanced_accuracy": 0.63,
                "dpd": 0.08,
                "equalized_odds_difference": 0.07,
            }
        ]
    )

    claim.to_csv(base / "cross_dataset_claim_summary_by_dataset.csv", index=False)
    svb.to_csv(base / "cross_dataset_subset_vs_baselines.csv", index=False)
    dummy.to_csv(base / "cross_dataset_best_by_dataset_model.csv", index=False)
    dummy.to_csv(base / "cross_dataset_selector_family_summary.csv", index=False)


def test_write_empirical_interpretation_section(tmp_path):
    cross_dir = tmp_path / "cross_dataset"
    output_path = tmp_path / "manuscript" / "empirical_interpretation.md"
    table_dir = tmp_path / "manuscript" / "generated_tables"

    write_fake_cross_dataset_outputs(cross_dir)

    cmd = [
        sys.executable,
        "scripts/write_empirical_interpretation.py",
        "--cross_dataset_dir",
        str(cross_dir),
        "--output_path",
        str(output_path),
        "--table_output_dir",
        str(table_dir),
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

    assert output_path.exists()
    assert (table_dir / "empirical_claim_summary_table.csv").exists()
    assert (table_dir / "subset_vs_fairness_baseline_table.csv").exists()

    text = output_path.read_text()

    assert "Empirical Results and Interpretation" in text
    assert "fairness-aware feature selection is not automatically proxy-leakage-aware" in text
    assert "Subset-aware FA-CMIM" in text
    assert "universal dominance claim" in text
    assert "universally best" not in text
    assert "TODO" not in text
