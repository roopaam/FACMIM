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
            }
        ]
    )

    best = pd.DataFrame(
        [
            {
                "dataset_key": "adult",
                "model": "gradient_boosting",
                "criterion": "lowest_joint_subset_mi",
                "selector": "SubsetFACMIM_k8_lambda1.0",
                "selector_family": "SubsetFA-CMIM",
                "accuracy": 0.83,
                "balanced_accuracy": 0.78,
                "joint_subset_mi_sensitive": 0.25,
                "sensitive_attacker_balanced_accuracy": 0.63,
                "dpd": 0.08,
                "equal_opportunity_difference": 0.07,
                "equalized_odds_difference": 0.09,
            }
        ]
    )

    all_results = pd.DataFrame(
        [
            {
                "dataset_key": "adult",
                "selector_family": "CMIM",
                "selector": "CMIM_k8",
                "model": "gradient_boosting",
                "accuracy": 0.84,
                "balanced_accuracy": 0.78,
                "dpd": 0.20,
                "equal_opportunity_difference": 0.18,
                "equalized_odds_difference": 0.22,
                "joint_subset_mi_sensitive": 0.59,
                "sensitive_attacker_balanced_accuracy": 0.86,
                "selected_features": "a|b|c",
            },
            {
                "dataset_key": "adult",
                "selector_family": "SubsetFA-CMIM",
                "selector": "SubsetFACMIM_k8_lambda1.0",
                "model": "gradient_boosting",
                "accuracy": 0.83,
                "balanced_accuracy": 0.77,
                "dpd": 0.08,
                "equal_opportunity_difference": 0.07,
                "equalized_odds_difference": 0.09,
                "joint_subset_mi_sensitive": 0.25,
                "sensitive_attacker_balanced_accuracy": 0.63,
                "selected_features": "a|e|f",
            },
        ]
    )

    claim.to_csv(base / "cross_dataset_claim_summary_by_dataset.csv", index=False)
    svb.to_csv(base / "cross_dataset_subset_vs_baselines.csv", index=False)
    best.to_csv(base / "cross_dataset_best_by_dataset_model.csv", index=False)
    all_results.to_csv(base / "cross_dataset_all_results.csv", index=False)


def write_fake_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal PNG signature plus placeholder bytes; enough for copying/reference tests.
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
    )


def test_generate_latex_artifacts(tmp_path):
    cross_dir = tmp_path / "cross_dataset"
    output_dir = tmp_path / "manuscript" / "latex"
    figure_root = tmp_path / "results" / "cross_dataset" / "plots"

    write_fake_cross_dataset_outputs(cross_dir)
    write_fake_png(figure_root / "cross_dataset_accuracy_vs_joint_mi.png")

    cmd = [
        sys.executable,
        "scripts/generate_latex_artifacts.py",
        "--cross_dataset_dir",
        str(cross_dir),
        "--output_dir",
        str(output_dir),
        "--figure_roots",
        str(figure_root),
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

    expected_tables = [
        "table_cross_dataset_claim_summary.tex",
        "table_subset_vs_fairness_baselines.tex",
        "table_best_proxy_leakage_by_dataset_model.tex",
        "table_selector_family_compact_summary.tex",
    ]

    for filename in expected_tables:
        path = output_dir / "tables" / filename
        assert path.exists(), filename

        text = path.read_text()
        assert r"\begin{table}" in text
        assert r"\toprule" in text
        assert r"\label{" in text

    refs_path = output_dir / "figure_references.tex"
    manifest_path = output_dir / "figure_manifest.csv"
    index_path = output_dir / "latex_artifacts_index.md"

    assert refs_path.exists()
    assert manifest_path.exists()
    assert index_path.exists()

    refs = refs_path.read_text()
    assert r"\includegraphics" in refs
    assert r"\caption" in refs
    assert r"\label{" in refs

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 1
