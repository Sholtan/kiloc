from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from kiloc.three_classifier import (
    ThreeClassCropClassifier,
    ThreeClassCropDataset,
    build_image_splits_from_train_records,
    build_balanced_sampler,
    count_by_class,
    evaluate_classifier,
    save_hardest_samples,
    train_classifier_one_epoch,
    write_prediction_csv,
)
from kiloc.utils.config import get_paths


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 3-class crop classifier study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/three_classifier/rexclude30_crop64.yaml"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def build_dataloader(
    *,
    dataset: ThreeClassCropDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: torch.utils.data.Sampler | None = None,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def build_scheduler(
    scheduler_cfg: dict[str, object],
    optimizer: torch.optim.Optimizer,
):
    if scheduler_cfg["name"] == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=scheduler_cfg["mode"],
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
            min_lr=scheduler_cfg["min_lr"],
        )
    if scheduler_cfg["name"] == "none":
        return None
    raise ValueError(f"Unsupported scheduler {scheduler_cfg['name']}")


def save_best_checkpoint(
    *,
    run_dir: Path,
    model: torch.nn.Module,
    epoch: int,
    previous_checkpoint: Path | None,
) -> Path:
    checkpoint_path = run_dir / f"three_classifier_best_f1_epoch_{epoch}.pth"
    if previous_checkpoint is not None and previous_checkpoint.exists() and previous_checkpoint != checkpoint_path:
        previous_checkpoint.unlink()
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def build_run_dir(checkpoint_root: Path, run_name: str | None) -> Path:
    base_dir = checkpoint_root / "three_classifier"
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = base_dir / run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)

    set_seed(int(cfg["seed"]))
    data_root, checkpoint_root = get_paths(device=cfg.get("paths_device", "h200"))
    run_dir = build_run_dir(checkpoint_root=checkpoint_root, run_name=args.run_name)
    shutil.copy(args.config, run_dir / "config.yaml")

    dataset_cfg = cfg["dataset"]
    crop_size = int(dataset_cfg["crop_size"])
    resize_hw = tuple(int(x) for x in dataset_cfg.get("resize_hw", [128, 128]))
    input_normalization = dataset_cfg.get("input_normalization", "imagenet")
    cache_images = bool(dataset_cfg.get("cache_images", False))
    split_cfg = dataset_cfg["image_split"]
    split_records, split_to_images = build_image_splits_from_train_records(
        annotated_train_csv=dataset_cfg["annotated_train_table"],
        mined_train_csv=dataset_cfg.get("mined_train_table"),
        seed=int(split_cfg["seed"]),
        train_ratio=float(split_cfg["train_ratio"]),
        validation_ratio=float(split_cfg["validation_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )

    split_counts = {split: count_by_class(records) for split, records in split_records.items()}
    with (run_dir / "dataset_summary.json").open("w") as handle:
        json.dump(
            {
                "crop_size": crop_size,
                "resize_hw": list(resize_hw),
                "source_mode": "original_train_only",
                "counts_by_split": split_counts,
                "annotated_train_table": dataset_cfg["annotated_train_table"],
                "mined_train_table": dataset_cfg.get("mined_train_table"),
                "image_split": split_cfg,
                "num_images_by_split": {
                    split: len(image_paths) for split, image_paths in split_to_images.items()
                },
            },
            handle,
            indent=2,
        )

    image_split_dir = run_dir / "image_splits"
    image_split_dir.mkdir(parents=True, exist_ok=True)
    for split, image_paths in split_to_images.items():
        with (image_split_dir / f"{split}_images.txt").open("w") as handle:
            for image_path in image_paths:
                handle.write(f"{image_path}\n")

    for split, counts in split_counts.items():
        missing_classes = [class_id for class_id, count in counts.items() if count == 0]
        if missing_classes:
            print(f"[three-classifier] split={split} missing classes={missing_classes}")

    datasets = {
        split: ThreeClassCropDataset(
            root=data_root,
            records=records,
            crop_size=crop_size,
            resize_hw=resize_hw,
            input_normalization=input_normalization,
            cache_images=cache_images,
        )
        for split, records in split_records.items()
    }

    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    balanced_sampling = bool(cfg.get("balanced_sampling", True))
    train_sampler = build_balanced_sampler(split_records["train"]) if balanced_sampling else None

    train_loader = build_dataloader(
        dataset=datasets["train"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not balanced_sampling,
        sampler=train_sampler,
        seed=int(cfg["seed"]),
    )
    val_loader = build_dataloader(
        dataset=datasets["validation"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=None,
        seed=int(cfg["seed"]),
    )
    test_loader = build_dataloader(
        dataset=datasets["test"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=None,
        seed=int(cfg["seed"]),
    )

    model = ThreeClassCropClassifier(
        pretrained=bool(cfg.get("is_pretrained", True)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        num_classes=3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[three-classifier] device={device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = build_scheduler(cfg["scheduler"], optimizer)

    history: list[dict[str, object]] = []
    best_f1_macro = -1.0
    best_epoch = -1
    best_checkpoint_path: Path | None = None

    for epoch in range(int(cfg["epochs"])):
        train_stats = train_classifier_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            dataloader=train_loader,
        )
        val_metrics, _ = evaluate_classifier(
            model=model,
            criterion=criterion,
            device=device,
            dataloader=val_loader,
        )

        if scheduler is not None:
            scheduler.step(float(val_metrics["f1_macro_present"]))

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_stats["train_loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_macro_present": val_metrics["f1_macro_present"],
            "val_f1_class_0": val_metrics["f1_class_0"],
            "val_f1_class_1": val_metrics["f1_class_1"],
            "val_f1_class_2": val_metrics["f1_class_2"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)

        if float(val_metrics["f1_macro_present"]) > best_f1_macro:
            best_f1_macro = float(val_metrics["f1_macro_present"])
            best_epoch = epoch + 1
            best_checkpoint_path = save_best_checkpoint(
                run_dir=run_dir,
                model=model,
                epoch=best_epoch,
                previous_checkpoint=best_checkpoint_path,
            )

        print(
            "Epoch "
            f"{epoch + 1}/{cfg['epochs']} | "
            f"train={train_stats['train_loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
            f"val_f1_macro_present={val_metrics['f1_macro_present']:.4f}"
        )

        with (run_dir / "history.json").open("w") as handle:
            json.dump(history, handle, indent=2)

    if best_checkpoint_path is None:
        raise RuntimeError("No best checkpoint was saved")

    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    split_results: dict[str, dict[str, object]] = {}
    hardest_cfg = cfg.get("hardest_samples", {})
    hardest_enabled = bool(hardest_cfg.get("enabled", False))
    hardest_splits = set(hardest_cfg.get("splits", ["validation", "test"]))

    for split, loader in (("validation", val_loader), ("test", test_loader)):
        metrics, rows = evaluate_classifier(
            model=model,
            criterion=criterion,
            device=device,
            dataloader=loader,
        )
        metrics["best_epoch"] = best_epoch
        metrics["best_checkpoint"] = best_checkpoint_path.name
        split_results[split] = metrics

        out_csv = run_dir / f"predictions_{split}.csv"
        write_prediction_csv(rows=rows, out_csv=out_csv)
        with (run_dir / f"{split}_results.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)

        if hardest_enabled and split in hardest_splits:
            save_hardest_samples(
                rows=rows,
                dataset=datasets[split],
                out_dir=run_dir / "hardest_samples" / split,
                ranking=str(hardest_cfg.get("ranking", "entropy")),
                top_k=int(hardest_cfg.get("top_k", 200)),
                export_images=bool(hardest_cfg.get("export_images", False)),
            )

    with (run_dir / "run_summary.json").open("w") as handle:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_checkpoint": best_checkpoint_path.name,
                "best_val_f1_macro_present": best_f1_macro,
                "device": str(device),
                "counts_by_split": split_counts,
                "validation": split_results["validation"],
                "test": split_results["test"],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
