import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.run_adult_pilot import make_adult_xy, make_synthetic_adult_like


def test_make_synthetic_adult_like_and_xy():
    df = make_synthetic_adult_like(n=100, seed=42)

    X, y, sensitive = make_adult_xy(df)

    assert len(X) == 100
    assert len(y) == 100
    assert len(sensitive) == 100
    assert "income" not in X.columns
    assert "sex" in X.columns
    assert y.isin([0, 1]).all()
    assert sensitive.name == "sex"


def test_adult_pilot_synthetic_smoke_runs(tmp_path):
    output_dir = tmp_path / "adult_smoke"

    cmd = [
        sys.executable,
        "scripts/run_adult_pilot.py",
        "--synthetic_smoke",
        "--sample_size",
        "250",
        "--k",
        "3",
        "--lambdas",
        "1.0",
        "--models",
        "logistic_regression",
        "--output_dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr

    output_path = output_dir / "adult_pilot_results.csv"
    metadata_path = output_dir / "adult_pilot_metadata.json"
    diagnostics_dir = output_dir / "diagnostics"

    assert output_path.exists()
    assert metadata_path.exists()
    assert diagnostics_dir.exists()

    results = pd.read_csv(output_path)

    assert not results.empty
    assert {"selector", "model", "accuracy", "dpd", "selected_features"}.issubset(results.columns)
    assert results["model"].eq("logistic_regression").all()
