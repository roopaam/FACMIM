from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_fairness_metrics,
    compute_leakage_metrics,
    compute_utility_metrics,
)


def main() -> None:
    rng = np.random.default_rng(42)

    y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1, 1, 1, 0, 0])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.6, 0.7, 0.3, 0.4])
    sensitive = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="sensitive")

    X_selected = pd.DataFrame(
        {
            "f_proxy": sensitive,
            "f_noise": rng.integers(0, 2, size=len(sensitive)),
        }
    )

    utility = compute_utility_metrics(y_true, y_pred, y_score)
    fairness = compute_fairness_metrics(y_true, y_pred, sensitive)
    leakage = compute_leakage_metrics(X_selected, sensitive, include_attacker=True)
    all_metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        sensitive=sensitive,
        X_selected=X_selected,
        include_attacker=True,
    )

    print("Utility:", utility)
    print("Fairness:", fairness)
    print("Leakage:", leakage)
    print("All:", all_metrics)


if __name__ == "__main__":
    main()
