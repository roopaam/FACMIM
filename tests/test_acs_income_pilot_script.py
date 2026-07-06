import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_acs_income_pilot_synthetic_smoke_runs(tmp_path):
    output_dir = tmp_path / "acs_income_smoke"

    cmd = [
        sys.executable,
        "scripts/run_acs_income_pilot.py",
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

    results_path = output_dir / "acs_income_pilot_results.csv"
    metadata_path = output_dir / "acs_income_pilot_metadata.json"

    assert results_path.exists()
    assert metadata_path.exists()

    df = pd.read_csv(results_path)

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


def test_acs_income_pilot_contains_all_selector_families(tmp_path):
    output_dir = tmp_path / "acs_income_smoke"

    cmd = [
        sys.executable,
        "scripts/run_acs_income_pilot.py",
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

    df = pd.read_csv(output_dir / "acs_income_pilot_results.csv")
    selectors = set(df["selector"])

    expected_prefixes = [
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

    for expected in expected_prefixes:
        assert expected in selectors


def test_acs_income_selected_features_do_not_include_sensitive_column(tmp_path):
    output_dir = tmp_path / "acs_income_smoke"

    cmd = [
        sys.executable,
        "scripts/run_acs_income_pilot.py",
        "--synthetic_smoke",
        "--sample_size",
        "300",
        "--k",
        "4",
        "--lambdas",
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

    df = pd.read_csv(output_dir / "acs_income_pilot_results.csv")

    for features in df["selected_features"]:
        selected = str(features).split("|")
        assert "sex" not in selected
