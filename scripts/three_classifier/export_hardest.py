from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from kiloc.three_classifier import (
    ThreeClassCropDataset,
    build_image_splits_from_train_records,
    save_hardest_samples,
)
from kiloc.utils.config import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hardest samples from a prediction CSV.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--ranking", choices=["entropy", "margin"], default="entropy")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--export-images", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = dict(row)
            for key in (
                "dataset_index",
                "true_label",
                "pred_label",
                "crop_size",
                "is_correct",
            ):
                parsed[key] = int(parsed[key])
            for key in (
                "x",
                "y",
                "prob_class_0",
                "prob_class_1",
                "prob_class_2",
                "entropy",
                "margin",
                "source_score",
            ):
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)

    data_root, _checkpoint_root = get_paths(device=cfg.get("paths_device", "h200"))
    dataset_cfg = cfg["dataset"]
    split_cfg = dataset_cfg["image_split"]
    split_records, _split_to_images = build_image_splits_from_train_records(
        annotated_train_csv=dataset_cfg["annotated_train_table"],
        mined_train_csv=dataset_cfg.get("mined_train_table"),
        seed=int(split_cfg["seed"]),
        train_ratio=float(split_cfg["train_ratio"]),
        validation_ratio=float(split_cfg["validation_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )
    records = split_records[args.split]
    dataset = ThreeClassCropDataset(
        root=data_root,
        records=records,
        crop_size=int(dataset_cfg["crop_size"]),
        resize_hw=tuple(int(x) for x in dataset_cfg.get("resize_hw", [128, 128])),
        input_normalization=dataset_cfg.get("input_normalization", "imagenet"),
        cache_images=bool(dataset_cfg.get("cache_images", False)),
    )

    rows = load_prediction_rows(args.predictions_csv)
    out_dir = args.out_dir or args.predictions_csv.parent / "hardest_samples" / args.split
    save_hardest_samples(
        rows=rows,
        dataset=dataset,
        out_dir=out_dir,
        ranking=args.ranking,
        top_k=args.top_k,
        export_images=args.export_images,
    )


if __name__ == "__main__":
    main()
