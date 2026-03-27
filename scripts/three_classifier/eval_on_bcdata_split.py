from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from kiloc.three_classifier import (
    ThreeClassCropClassifier,
    ThreeClassCropDataset,
    evaluate_classifier,
    load_split_records,
    resolve_classifier_checkpoint,
    write_prediction_csv,
)
from kiloc.utils.config import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 3-class crop classifier on a real BCData split."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bcdata-split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--annotated-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.run_dir / "config.yaml"
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)

    data_root, _checkpoint_root = get_paths(device=cfg.get("paths_device", "h200"))
    dataset_cfg = cfg["dataset"]
    records = load_split_records(
        split=args.bcdata_split,
        annotated_csv=args.annotated_csv or Path("cells_tables") / f"{args.bcdata_split}_points.csv",
        mined_csv=None,
    )

    dataset = ThreeClassCropDataset(
        root=data_root,
        records=records,
        crop_size=int(dataset_cfg["crop_size"]),
        resize_hw=tuple(int(x) for x in dataset_cfg.get("resize_hw", [128, 128])),
        input_normalization=dataset_cfg.get("input_normalization", "imagenet"),
        cache_images=bool(args.cache_images or dataset_cfg.get("cache_images", False)),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)) > 0,
    )

    checkpoint_path = resolve_classifier_checkpoint(args.run_dir, args.checkpoint)
    model = ThreeClassCropClassifier(
        pretrained=bool(cfg.get("is_pretrained", True)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        num_classes=3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    metrics, rows = evaluate_classifier(
        model=model,
        criterion=criterion,
        device=device,
        dataloader=dataloader,
    )
    metrics["checkpoint"] = checkpoint_path.name
    metrics["bcdata_split"] = args.bcdata_split
    metrics["annotated_csv"] = (args.annotated_csv or Path("cells_tables") / f"{args.bcdata_split}_points.csv").as_posix()
    metrics["num_records"] = len(records)

    out_dir = args.out_dir or (args.run_dir / "external_bcdata_eval" / args.bcdata_split)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(rows=rows, out_csv=out_dir / "predictions.csv")
    with (out_dir / "results.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print((out_dir / "predictions.csv").as_posix())
    print((out_dir / "results.json").as_posix())


if __name__ == "__main__":
    main()
