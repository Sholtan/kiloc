from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from kiloc.three_classifier.visualization import (
    RANKED_CATEGORY_SPECS,
    load_prediction_rows,
    save_ranked_full_image_examples,
)
from kiloc.utils.config import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save full-resolution images for selected high-scoring cross-class crop examples."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(RANKED_CATEGORY_SPECS),
        default=list(RANKED_CATEGORY_SPECS),
        help="Which ranked categories to export.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of full-image examples to save per category.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=196,
        help="Rectangle size in input-image pixels.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.run_dir / "config.yaml"
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)

    data_root, _checkpoint_root = get_paths(device=cfg.get("paths_device", "h200"))
    predictions_csv = args.predictions_csv or (args.run_dir / f"predictions_{args.split}.csv")
    rows = load_prediction_rows(predictions_csv)

    out_dir = args.out_dir or (
        args.run_dir / "crossclass_full_image_examples" / args.split
    )
    summary = save_ranked_full_image_examples(
        rows=rows,
        data_root=data_root,
        out_dir=out_dir,
        category_slugs=list(args.categories),
        top_k=int(args.top_k),
        box_size=int(args.box_size),
    )
    print((out_dir / "summary.json").as_posix())
    for slug, info in summary["categories"].items():
        print(f"{slug}: saved={info['top_k_saved']} dir={info['image_dir']}")


if __name__ == "__main__":
    main()
