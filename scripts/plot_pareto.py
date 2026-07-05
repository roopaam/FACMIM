from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.summary import load_results, plot_all_pareto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Pareto frontier plots.")

    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--utility_col", type=str, default="accuracy")
    parser.add_argument(
        "--fairness_cols",
        nargs="+",
        default=[
            "dpd",
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "joint_subset_mi_sensitive",
        ],
    )
    parser.add_argument("--group_col", type=str, default="model")
    parser.add_argument("--label_col", type=str, default="selector")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_results(args.results_path)

    paths = plot_all_pareto(
        df,
        output_dir=args.output_dir,
        utility_col=args.utility_col,
        fairness_cols=args.fairness_cols,
        group_col=args.group_col,
        label_col=args.label_col,
    )

    print("Saved Pareto plots:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
