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

from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.losses.losses import SigmoidSumHuber, SigmoidWeightedMSE, sigmoid_focal_loss
from kiloc.mining import (
    BCDataHardNegativeDataset,
    HardNegativeWeightMapGenerator,
    build_hardneg_loss,
    collate_fn_hardneg,
    group_mined_false_positives_by_image,
    read_mined_false_positive_csv,
    summarize_grouped_mined_false_positives,
    train_one_epoch_hardneg,
)
from kiloc.model.kiloc_net import KiLocNet
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.training.checkpoints import save_single_best_checkpoint
from kiloc.training.ema import ModelEMA
from kiloc.training.train import val_one_epoch
from kiloc.utils.config import get_paths


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


def build_criterion(cfg: dict):
    if cfg["loss"] == "sigmoid_weighted_mse_loss":
        detection_loss = SigmoidWeightedMSE(
            alpha_pos=cfg["alpha_pos"],
            alpha_neg=cfg["alpha_neg"],
            q=cfg["q"],
        )
    elif cfg["loss"] == "sigmoid_focal_loss":
        detection_loss = sigmoid_focal_loss
    else:
        raise ValueError(
            "loss must be one of {'sigmoid_weighted_mse_loss', 'sigmoid_focal_loss'}"
        )

    if cfg.get("count_loss", False):
        sum_huber = SigmoidSumHuber()

        def criterion(pred, target, pos_pts_tuple, neg_pts_tuple):
            det = detection_loss(pred, target, pos_pts_tuple, neg_pts_tuple)
            cnt = sum_huber(pred, target, pos_pts_tuple, neg_pts_tuple)
            return det + cfg["lambda_count"] * cnt

    else:
        criterion = detection_loss

    if cfg.get("suppression_loss", False):
        from kiloc.losses.losses import SigmoidOppositeSuppression

        suppression = SigmoidOppositeSuppression(weight=cfg.get("lambda_suppression", 1.0))
        previous_criterion = criterion

        def criterion(pred, target, pos_pts_tuple, neg_pts_tuple):
            base = previous_criterion(pred, target, pos_pts_tuple, neg_pts_tuple)
            sup = suppression(pred, target, pos_pts_tuple, neg_pts_tuple)
            return base + sup

    return criterion


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    sch_cfg = cfg["scheduler"]
    if sch_cfg["name"] == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=sch_cfg["mode"],
            factor=sch_cfg["factor"],
            patience=sch_cfg["patience"],
            min_lr=sch_cfg["min_lr"],
        )
    if sch_cfg["name"] == "none":
        return None
    raise ValueError("scheduler must be ReduceLROnPlateau or none")


def resolve_threshold(cfg: dict) -> tuple[float, float]:
    base_threshold = cfg.get("threshold", 0.5)
    return (
        float(cfg.get("thr_pos", base_threshold)),
        float(cfg.get("thr_neg", base_threshold)),
    )


def resolve_path(path_value: str | Path, *, relative_to: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (relative_to / path).resolve()


def load_merged_config(config_path: Path) -> tuple[dict, Path, dict]:
    with config_path.open("r", encoding="utf-8") as f:
        finetune_cfg = yaml.safe_load(f)

    if "base_config" not in finetune_cfg:
        raise KeyError("Finetune config must include 'base_config'")

    base_config_path = resolve_path(finetune_cfg["base_config"], relative_to=config_path.parent)
    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    merged_cfg = dict(base_cfg)
    for key, value in finetune_cfg.items():
        if key == "base_config":
            continue
        merged_cfg[key] = value

    merged_cfg["_base_config_path"] = str(base_config_path)
    return merged_cfg, base_config_path, finetune_cfg


def build_run_dir(
    checkpoint_dir: Path,
    *,
    out_dir: str | None,
    run_name: str | None,
) -> Path:
    if out_dir:
        checkpoint_dir = checkpoint_dir / out_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        run_dir = checkpoint_dir / run_name
    else:
        run_dir = checkpoint_dir / f"hardneg_finetune_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a localization model with mined hard-negative suppression loss."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", default="hardneg_finetune")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device-key", default="h200", choices=["hpvictus", "collab", "h200"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, base_config_path, _ = load_merged_config(args.config)

    if cfg.get("augmentation", False):
        raise ValueError(
            "Hard-negative fine-tuning v1 requires augmentation=false so mined centers stay aligned."
        )
    if "base_checkpoint" not in cfg:
        raise KeyError("Finetune config must include 'base_checkpoint'")
    if "hardneg_csv" not in cfg:
        raise KeyError("Finetune config must include 'hardneg_csv'")
    if "lambda_hardneg" not in cfg:
        raise KeyError("Finetune config must include 'lambda_hardneg'")

    base_checkpoint_path = resolve_path(cfg["base_checkpoint"], relative_to=args.config.parent)
    hardneg_csv_path = resolve_path(cfg["hardneg_csv"], relative_to=args.config.parent)

    set_seed(int(cfg["seed"]))

    root_dir, checkpoint_dir = get_paths(device=args.device_key)
    run_dir = build_run_dir(
        checkpoint_dir=checkpoint_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
    )
    print(f"RUN_DIR:{run_dir}")

    shutil.copy(args.config, run_dir / "finetune_config.yaml")
    shutil.copy(base_config_path, run_dir / "base_config.yaml")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not str(k).startswith("_")}, f, sort_keys=False)

    threshold = resolve_threshold(cfg)
    print(f"Validation thresholds: thr_pos={threshold[0]:.3f}, thr_neg={threshold[1]:.3f}")

    heatmap_gen = LocHeatmap(
        out_hw=tuple(cfg["out_hw"]),
        in_hw=tuple(cfg["in_hw"]),
        sigma=cfg["sigma"],
        dtype=torch.float32,
    )

    base_dataset_train = BCDataDataset(
        root=root_dir,
        split="train",
        target_transform=heatmap_gen,
        input_normalization=cfg["input_normalization"],
    )
    dataset_val = BCDataDataset(
        root=root_dir,
        split="validation",
        target_transform=heatmap_gen,
        input_normalization=cfg["input_normalization"],
    )

    mined_rows = read_mined_false_positive_csv(hardneg_csv_path)
    hardneg_by_image = group_mined_false_positives_by_image(
        mined_rows,
        min_score=cfg.get("hardneg_min_score", None),
        pred_classes=cfg.get("hardneg_pred_classes", None),
        max_points_per_image=cfg.get("hardneg_max_points_per_image", None),
    )
    if not hardneg_by_image:
        raise RuntimeError("No mined hard-negative candidates remained after filtering.")

    train_image_ids = {image_path.stem for image_path, _, _ in base_dataset_train.samples}
    extra_image_ids = sorted(set(hardneg_by_image).difference(train_image_ids))
    if extra_image_ids:
        print(
            f"Warning: dropping {len(extra_image_ids)} mined image ids not present in train split. "
            f"Examples: {extra_image_ids[:5]}"
        )
        hardneg_by_image = {
            image_id: image_rows
            for image_id, image_rows in hardneg_by_image.items()
            if image_id in train_image_ids
        }

    hardneg_summary = summarize_grouped_mined_false_positives(hardneg_by_image)
    hardneg_summary["hardneg_csv"] = str(hardneg_csv_path)
    hardneg_summary["images_without_candidates"] = len(train_image_ids) - len(hardneg_by_image)
    hardneg_summary["hardneg_min_score"] = cfg.get("hardneg_min_score", None)
    hardneg_summary["hardneg_pred_classes"] = cfg.get("hardneg_pred_classes", None)
    hardneg_summary["hardneg_max_points_per_image"] = cfg.get("hardneg_max_points_per_image", None)
    with (run_dir / "hardneg_candidate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(hardneg_summary, f, indent=2)

    hardneg_target_generator = HardNegativeWeightMapGenerator(
        out_hw=tuple(cfg["out_hw"]),
        in_hw=tuple(cfg["in_hw"]),
        sigma=float(cfg.get("hardneg_sigma", 2.0)),
        dtype=torch.float32,
        aggregation=str(cfg.get("hardneg_aggregation", "max")),
        use_score_weights=bool(cfg.get("hardneg_use_score_weights", False)),
        constant_weight=float(cfg.get("hardneg_constant_weight", 1.0)),
        clamp_max=cfg.get("hardneg_clamp_max", None),
    )
    dataset_train = BCDataHardNegativeDataset(
        base_dataset_train,
        hardneg_by_image=hardneg_by_image,
        hardneg_target_transform=hardneg_target_generator,
    )

    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    persistent_workers = num_workers > 0

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn_hardneg,
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
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        generator=g,
        persistent_workers=persistent_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KiLocNet(pretrained=False, backbone_name=cfg["backbone"]).to(device)
    model.load_state_dict(torch.load(base_checkpoint_path, map_location="cpu"))

    base_criterion = build_criterion(cfg)
    hardneg_loss_name = str(cfg.get("hardneg_loss_name", "local_bce_logits"))
    hardneg_gce_q = float(cfg.get("hardneg_gce_q", 0.7))
    hardneg_criterion = build_hardneg_loss(
        hardneg_loss_name,
        gce_q=hardneg_gce_q,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = build_scheduler(cfg, optimizer)

    use_ema = cfg.get("ema", False)
    ema = None
    ema_start_epoch = cfg.get("ema_start_epoch", 5)
    lambda_hardneg = float(cfg["lambda_hardneg"])

    best_f1_macro = -1.0
    best_epoch = -1
    best_checkpoint_path: Path | None = None
    history: list[dict[str, float | int]] = []

    initial_val_result = val_one_epoch(
        model=model,
        criterion=base_criterion,
        device=device,
        val_loader=dataloader_val,
        kernel_size=cfg["kernel_size"],
        threshold=threshold,
        merge_radius=cfg["merge_radius"],
        matching_radius=cfg["matching_radius"],
        tta=cfg.get("tta", False),
    )
    (
        initial_val_loss,
        initial_precision,
        initial_recall,
        initial_f1,
        initial_precision_pos,
        initial_recall_pos,
        initial_f1_pos,
        initial_precision_neg,
        initial_recall_neg,
        initial_f1_neg,
        initial_f1_macro,
    ) = initial_val_result
    initial_validation = {
        "val_loss": initial_val_loss,
        "precision": initial_precision,
        "recall": initial_recall,
        "f1": initial_f1,
        "precision_pos": initial_precision_pos,
        "recall_pos": initial_recall_pos,
        "f1_pos": initial_f1_pos,
        "precision_neg": initial_precision_neg,
        "recall_neg": initial_recall_neg,
        "f1_neg": initial_f1_neg,
        "f1_macro": initial_f1_macro,
    }
    with (run_dir / "initial_validation.json").open("w", encoding="utf-8") as f:
        json.dump(initial_validation, f, indent=2)
    print(
        f"Initial validation | val={initial_val_loss:.4f} | "
        f"P={initial_precision:.3f}, R={initial_recall:.3f}, F1_macro={initial_f1_macro:.3f}"
    )

    for epoch_idx in range(int(cfg["epochs"])):
        if use_ema and ema is None and epoch_idx == ema_start_epoch:
            ema = ModelEMA(
                model=model,
                decay=cfg.get("ema_decay", 0.999),
                device=cfg.get("ema_device", None),
            )
            print(f"EMA initialized at epoch {epoch_idx + 1}")

        train_stats = train_one_epoch_hardneg(
            model=model,
            base_criterion=base_criterion,
            hardneg_criterion=hardneg_criterion,
            lambda_hardneg=lambda_hardneg,
            optimizer=optimizer,
            device=device,
            trainloader=dataloader_train,
            ema=ema,
        )

        eval_model = ema.module if ema is not None else model
        val_result = val_one_epoch(
            model=eval_model,
            criterion=base_criterion,
            device=device,
            val_loader=dataloader_val,
            kernel_size=cfg["kernel_size"],
            threshold=threshold,
            merge_radius=cfg["merge_radius"],
            matching_radius=cfg["matching_radius"],
            tta=cfg.get("tta", False),
        )
        (
            val_loss,
            precision,
            recall,
            f1,
            precision_pos,
            recall_pos,
            f1_pos,
            precision_neg,
            recall_neg,
            f1_neg,
            f1_macro,
        ) = val_result

        print(
            f"Epoch {epoch_idx + 1}/{cfg['epochs']} | "
            f"train_total={train_stats['train_loss']:.4f} | "
            f"train_base={train_stats['train_base_loss']:.4f} | "
            f"train_hardneg={train_stats['train_hardneg_loss']:.4f} | "
            f"train_hardneg_raw={train_stats['train_hardneg_loss_raw']:.4f} | "
            f"hardneg/base={train_stats['train_hardneg_to_base_ratio']:.3f} | "
            f"val={val_loss:.4f} | "
            f"F1_macro={f1_macro:.3f}"
        )

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_epoch = epoch_idx + 1
            best_checkpoint_path = save_single_best_checkpoint(
                run_dir=run_dir,
                best_epoch=best_epoch,
                model=model,
                ema_module=ema.module if ema is not None else None,
                previous_checkpoint_path=best_checkpoint_path,
            )

        history_entry = {
            "epoch": epoch_idx + 1,
            "train_loss": train_stats["train_loss"],
            "train_base_loss": train_stats["train_base_loss"],
            "train_hardneg_loss": train_stats["train_hardneg_loss"],
            "train_hardneg_loss_raw": train_stats["train_hardneg_loss_raw"],
            "train_hardneg_to_base_ratio": train_stats["train_hardneg_to_base_ratio"],
            "hardneg_weight_sum_total": train_stats["hardneg_weight_sum_total"],
            "hardneg_active_images": train_stats["hardneg_active_images"],
            "num_images": train_stats["num_images"],
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_pos": precision_pos,
            "recall_pos": recall_pos,
            "f1_pos": f1_pos,
            "precision_neg": precision_neg,
            "recall_neg": recall_neg,
            "f1_neg": f1_neg,
            "f1_macro": f1_macro,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(history_entry)
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if scheduler is not None:
            scheduler.step(f1_macro)

    if best_checkpoint_path is None:
        raise RuntimeError("Fine-tuning finished without saving a best checkpoint.")

    metadata = {
        "run_dir": str(run_dir),
        "base_config": str(base_config_path),
        "base_checkpoint": str(base_checkpoint_path),
        "hardneg_csv": str(hardneg_csv_path),
        "initial_val_f1_macro": initial_f1_macro,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1_macro,
        "best_checkpoint": best_checkpoint_path.name,
        "lambda_hardneg": lambda_hardneg,
        "hardneg_loss_name": hardneg_loss_name,
        "hardneg_gce_q": hardneg_gce_q,
        "hardneg_sigma": float(cfg.get("hardneg_sigma", 2.0)),
        "hardneg_aggregation": str(cfg.get("hardneg_aggregation", "max")),
        "hardneg_use_score_weights": bool(cfg.get("hardneg_use_score_weights", False)),
        "hardneg_constant_weight": float(cfg.get("hardneg_constant_weight", 1.0)),
    }
    with (run_dir / "finetune_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"BEST EPOCH WAS: {best_epoch}")
    print(f"BEST CHECKPOINT: {best_checkpoint_path.name}")


if __name__ == "__main__":
    main()
