from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiloc.three_classifier.filter_eval import (
    build_default_tau_grid,
    evaluate_filter_on_localization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 3-class crop classifier as a reject-only filter on a localization run.",
    )
    parser.add_argument("--localization-run-dir", type=Path, required=True)
    parser.add_argument("--localization-checkpoint", type=str, default=None)
    parser.add_argument("--classifier-run-dir", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument(
        "--tau-grid",
        type=float,
        nargs="*",
        default=None,
        help="Optional explicit tau values. Default is 0.00, 0.01, ..., 1.00.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tau_grid = args.tau_grid if args.tau_grid is not None and len(args.tau_grid) > 0 else build_default_tau_grid()
    summary = evaluate_filter_on_localization(
        localization_run_dir=args.localization_run_dir,
        localization_checkpoint=args.localization_checkpoint,
        classifier_run_dir=args.classifier_run_dir,
        classifier_checkpoint=args.classifier_checkpoint,
        split=args.split,
        tau_grid=tau_grid,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
