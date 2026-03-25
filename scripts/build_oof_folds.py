from __future__ import annotations

import argparse
from pathlib import Path

from kiloc.oof.folds import (
    build_balanced_image_folds,
    scan_bcdata_images,
    summarize_fold_assignments,
    write_fold_manifests,
)
from kiloc.utils.config import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic image-level OOF folds for BCData."
    )
    parser.add_argument(
        "--device",
        default="h200",
        choices=["hpvictus", "collab", "h200"],
        help="Device key used by kiloc.utils.config.get_paths().",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to partition into folds.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds to build.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=41,
        help="Seed used for deterministic tie-breaking before fold assignment.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for fold manifests. Defaults to splits/oof_<split>_<k>fold_seed<seed>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root, _ = get_paths(device=args.device)
    records = scan_bcdata_images(data_root=data_root, split=args.split)
    assignments = build_balanced_image_folds(
        records=records,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("splits") / f"oof_{args.split}_{args.num_folds}fold_seed{args.seed}"

    write_fold_manifests(assignments=assignments, out_dir=out_dir, num_folds=args.num_folds)

    print(f"Scanned {len(records)} images from split={args.split!r}")
    for row in summarize_fold_assignments(assignments, num_folds=args.num_folds):
        print(
            f"fold={row['fold']} | "
            f"images={row['num_images']} | "
            f"pos={row['num_pos']} | "
            f"neg={row['num_neg']} | "
            f"cells={row['num_cells']}"
        )
    print(f"Saved fold manifests to: {out_dir}")


if __name__ == "__main__":
    main()
