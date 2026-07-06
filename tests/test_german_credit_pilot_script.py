import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_german_credit_smoke(output_dir: Path):
    cmd = [
        sys.executable,
        "scripts/run_german_credit_pilot.py",
        "--synthetic_smoke",
        "--sample_size",
        "300",
        "--k",
        "4",
        "--lambdas",
        "0.0",
        "1.0",
        "--models",
        "logistic_regression",
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

    return pd.read_csv(output_dir / "german_credit_pilot_results.csv")


def test_german_credit_pilot_synthetic_smoke_runs(tmp_path):
    output_dir = tmp_path / "german_credit_smoke"

    df = run_german_credit_smoke(output_dir)

    results_path = output_dir / "german_credit_pilot_results.csv"
    metadata_path = output_dir / "german_credit_pilot_metadata.json"

    assert results_path.exists()
    assert metadata_path.exists()

    # 2 non-lambda selectors + 6 lambda selectors * 2 lambdas = 14 rows
    assert len(df) == 14

    required_cols = {
        "dataset",
        "selector",
        "model",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auroc",
        "dpd",
        "dpr",
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "mean_selected_mi_sensitive",
        "joint_subset_mi_sensitive",
        "sensitive_attacker_balanced_accuracy",
        "selected_features",
    }

    assert required_cols.issubset(df.columns)


def test_german_credit_pilot_contains_all_selector_families(tmp_path):
    output_dir = tmp_path / "german_credit_smoke"

    df = run_german_credit_smoke(output_dir)
    selectors = set(df["selector"])

    expected_selectors = [
        "CMIM_k4",
        "mRMR_k4",
        "ProxyRank_k4_lambda0.0",
        "ProxyRank_k4_lambda1.0",
        "FairmRMR_k4_lambda0.0",
        "FairmRMR_k4_lambda1.0",
        "FairCFS_k4_lambda0.0",
        "FairCFS_k4_lambda1.0",
        "FairLasso_k4_lambda0.0",
        "FairLasso_k4_lambda1.0",
        "BasicFACMIM_k4_lambda0.0",
        "BasicFACMIM_k4_lambda1.0",
        "SubsetFACMIM_k4_lambda0.0",
        "SubsetFACMIM_k4_lambda1.0",
    ]

    for selector in expected_selectors:
        assert selector in selectors


def test_german_credit_selected_features_do_not_include_age_sensitive_fields(tmp_path):
    output_dir = tmp_path / "german_credit_smoke"

    df = run_german_credit_smoke(output_dir)

    for features in df["selected_features"]:
        selected = str(features).split("|")
        assert "age" not in selected
        assert "age_group" not in selected
