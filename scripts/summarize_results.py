from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.summary import save_summary_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")

    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--utility_col", type=str, default="accuracy")
    parser.add_argument("--fairness_col", type=str, default="dpd")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = save_summary_outputs(
        results_path=args.results_path,
        output_dir=args.output_dir,
        utility_col=args.utility_col,
        fairness_col=args.fairness_col,
    )

    print("Saved summary outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
