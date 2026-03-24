"""Run the full train/val loop for point-centered classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from kiloc.datasets import BCDataPointDataset
from kiloc.model.classifier_net import ResNetClassifier
from kiloc.training.ema import ModelEMA
from kiloc.utils.config import get_paths

from kiloc.training.train_classifier import train_one_epoch, validate_one_epoch



def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_background_rgb(
    background_stats_path: str | Path | None,
) -> tuple[int, int, int]:
    if background_stats_path is None:
        raise ValueError("background_stats_path is None")

    path = Path(background_stats_path)
    if not path.exists():
        raise ValueError(f"{background_stats_path} doesn't exist")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "median_rgb_uint8" not in data:
        raise ValueError(f"no median_rgb_uint8 in {background_stats_path}")

    rgb = data["median_rgb_uint8"]
    if not isinstance(rgb, list) or len(rgb) != 3:
        raise ValueError(f"error when reading rgb: {rgb}")

    return tuple(int(v) for v in rgb)


def count_labels_in_csv(points_csv: str | Path) -> tuple[int, int]:
    import csv

    num_pos = 0
    num_neg = 0
    with Path(points_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError(f"CSV must contain 'label' column: {points_csv}")
        for row in reader:
            label = int(row["label"])
            if label == 1:
                num_pos += 1
            elif label == 0:
                num_neg += 1
            else:
                raise ValueError(
                    f"Labels must be binary 0/1, got label={label} in {points_csv}"
                )
    return num_pos, num_neg


def build_train_transform(enable_augmentation: bool) -> A.Compose | None:
    if not enable_augmentation:
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.10,
                rotate_limit=0,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.12,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=6,
                        sat_shift_limit=10,
                        val_shift_limit=8,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.GaussNoise(std_range=(0.005, 0.015), mean_range=(0.0, 0.0), p=1.0),
                    A.ImageCompression(quality_range=(85, 100), p=1.0),
                ],
                p=0.15,
            ),
        ]
    )



def main(config_path: str, run_suffix: str | None, out_dir: str | None, run_name: str | None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    seed = cfg["seed"]

    backbone = cfg["backbone"]
    crop_size = cfg["crop_size"]
    input_size = cfg["input_size"]
    is_pretrained = cfg["is_pretrained"]
    input_normalization = cfg.get("input_normalization", "imagenet")
    dropout = cfg.get("dropout", 0.0)
    adapt_small_inputs = cfg.get("adapt_small_inputs", True)
    threshold = cfg.get("threshold", 0.5)
    use_amp = cfg.get("use_amp", True)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    set_seed(seed)

    root_dir, checkpoint_dir = get_paths(device=cfg.get("device_key", "h200"))
    if out_dir:
        checkpoint_dir = checkpoint_dir / out_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        run_dir = checkpoint_dir / run_name
    elif run_suffix:
        run_dir = checkpoint_dir / f"run_{run_suffix}_{timestamp}"
    else:
        run_dir = checkpoint_dir / f"run_{timestamp}"

    run_dir.mkdir(parents=True)
    print(f"RUN_DIR:{run_dir}")

    shutil.copy(config_path, run_dir / "config.yaml")

    background_rgb = read_background_rgb(
        background_stats_path=cfg["background_stats_path"],
    )
    print(f"Using background_rgb={background_rgb}")

    train_points_csv = Path(cfg.get("train_points_csv", "bcdata_point_tables/train_points.csv"))
    val_points_csv = Path(cfg.get("val_points_csv", "bcdata_point_tables/validation_points.csv"))

    train_tf = build_train_transform(cfg.get("augmentation", False))
    if train_tf is not None:
        print("Augmentations will be applied to the train set")
    else:
        print("No augmentations will be applied")

    print(f"Using {input_normalization} input normalization")
    print(f"Crop size: {crop_size}, input size: {input_size}, backbone: {backbone}")

    dataset_train = BCDataPointDataset(
        data_root=root_dir,
        points_csv=train_points_csv,
        crop_size=crop_size,
        resize_to=input_size,
        background_rgb=background_rgb,
        image_transform=train_tf,
        input_normalization=input_normalization,
    )
    dataset_val = BCDataPointDataset(
        data_root=root_dir,
        points_csv=val_points_csv,
        crop_size=crop_size,
        resize_to=input_size,
        background_rgb=background_rgb,
        image_transform=None,
        input_normalization=input_normalization,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    persistent_workers = num_workers > 0

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=BCDataPointDataset.collate_fn,
        pin_memory=True,
        drop_last=False,
        generator=g,
        persistent_workers=persistent_workers,
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=BCDataPointDataset.collate_fn,
        pin_memory=True,
        drop_last=False,
        generator=g,
        persistent_workers=persistent_workers,
    )

    model = ResNetClassifier(
        backbone_name=backbone,
        input_size=input_size,
        num_classes=1,
        pretrained=is_pretrained,
        dropout=dropout,
        adapt_small_inputs=adapt_small_inputs,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    use_pos_weight = cfg.get("use_pos_weight", True)
    if use_pos_weight:
        if "pos_weight" in cfg and cfg["pos_weight"] not in (None, "auto"):
            pos_weight_value = float(cfg["pos_weight"])
        else:
            num_pos, num_neg = count_labels_in_csv(train_points_csv)
            if num_pos == 0:
                raise ValueError("No positive samples found in training CSV.")
            pos_weight_value = num_neg / num_pos

        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_value:.6f}")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss without pos_weight")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    sch_cfg = cfg.get("scheduler", {"name": "none"})
    if sch_cfg["name"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=sch_cfg["mode"],
            factor=sch_cfg["factor"],
            patience=sch_cfg["patience"],
            min_lr=sch_cfg["min_lr"],
        )
    elif sch_cfg["name"] == "none":
        scheduler = None
    else:
        raise ValueError("scheduler must be ReduceLROnPlateau or none")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    use_ema = cfg.get("ema", False)
    ema = None
    ema_start_epoch = cfg.get("ema_start_epoch", 5)

    history: list[dict] = []
    best_f1_macro = -1.0
    best_epoch = -1

    for i in range(epochs):
        if use_ema and ema is None and i == ema_start_epoch:
            ema = ModelEMA(
                model=model,
                decay=cfg.get("ema_decay", 0.999),
                device=cfg.get("ema_device", None),
            )
            print(f"EMA initialized at epoch {i + 1}")

        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            trainloader=dataloader_train,
            ema=ema,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
        )

        eval_model = ema.module if ema is not None else model

        val_loss, val_metrics = validate_one_epoch(
            model=eval_model,
            criterion=criterion,
            device=device,
            val_loader=dataloader_val,
            threshold=threshold,
            use_amp=use_amp,
        )

        print(
            f"Epoch {i + 1}/{epochs} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"P={val_metrics['precision']:.3f}, "
            f"R={val_metrics['recall']:.3f}, "
            f"F1_micro={val_metrics['f1']:.3f}, "
            f"F1_macro={val_metrics['f1_macro']:.3f}"
        )

        if val_metrics["f1_macro"] > best_f1_macro:
            best_f1_macro = val_metrics["f1_macro"]
            best_epoch = i + 1
            torch.save(model.state_dict(), run_dir / "classifier_epoch_best.pth")
            if ema is not None:
                torch.save(ema.module.state_dict(), run_dir / "classifier_epoch_best_ema.pth")

        if i == epochs - 1:
            torch.save(model.state_dict(), run_dir / "classifier_epoch_last.pth")
            if ema is not None:
                torch.save(ema.module.state_dict(), run_dir / "classifier_epoch_last_ema.pth")

        history.append(
            {
                "epoch": i + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
                "accuracy": val_metrics["accuracy"],
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "precision_pos": val_metrics["precision_pos"],
                "recall_pos": val_metrics["recall_pos"],
                "f1_pos": val_metrics["f1_pos"],
                "precision_neg": val_metrics["precision_neg"],
                "recall_neg": val_metrics["recall_neg"],
                "f1_neg": val_metrics["f1_neg"],
                "f1_macro": val_metrics["f1_macro"],
                "tp": val_metrics["tp"],
                "fp": val_metrics["fp"],
                "fn": val_metrics["fn"],
                "tn": val_metrics["tn"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if scheduler is not None:
            scheduler.step(val_metrics["f1_macro"])

    best_path = run_dir / "classifier_epoch_best.pth"
    if best_path.exists():
        best_path.rename(run_dir / f"classifier_best_f1_epoch_{best_epoch}.pth")

    best_path_ema = run_dir / "classifier_epoch_best_ema.pth"
    if best_path_ema.exists():
        best_path_ema.rename(run_dir / f"classifier_best_f1_epoch_{best_epoch}_ema.pth")

    print(f"BEST EPOCH WAS: {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/context_radius_study/train_classifier.yaml",
    )
    parser.add_argument("--run_suffix", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    main(args.config, args.run_suffix, args.out_dir, args.run_name)