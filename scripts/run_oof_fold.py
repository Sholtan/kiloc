from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from kiloc.datasets.bcdata import (
    AlbumentationsJointTransform,
    BCDataDataset,
    collate_fn,
)
from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.evaluation.metrics import compute_metrics
from kiloc.evaluation.tta import tta_forward
from kiloc.losses.losses import SigmoidSumHuber, SigmoidWeightedMSE, sigmoid_focal_loss
from kiloc.model.kiloc_net import KiLocNet
from kiloc.oof import (
    build_raw_prediction_rows,
    load_image_ids,
    relation_artifact_paths,
    write_raw_prediction_csv,
)
from kiloc.oof import (
    build_relation_rows,
    load_gt_by_image_from_image_paths,
    summarize_relation_rows,
    write_relation_csv,
    write_relation_summary,
)
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.training.checkpoints import save_single_best_checkpoint
from kiloc.training.ema import ModelEMA
from kiloc.training.train import train_one_epoch, val_one_epoch
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


def build_train_transform() -> AlbumentationsJointTransform:
    transform = A.Compose(
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
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    return AlbumentationsJointTransform(transform)


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

        suppression = SigmoidOppositeSuppression(
            weight=cfg.get("lambda_suppression", 1.0)
        )
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


def build_run_dir(
    checkpoint_dir: Path,
    out_dir: str | None,
    run_name: str | None,
    fold_index: int,
) -> Path:
    if out_dir:
        checkpoint_dir = checkpoint_dir / out_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        run_dir = checkpoint_dir / run_name
    else:
        run_dir = checkpoint_dir / f"oof_fold_{fold_index}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def infer_holdout(
    *,
    model: KiLocNet,
    loader: DataLoader,
    dataset: BCDataDataset,
    device: torch.device | str,
    fold_index: int,
    threshold: tuple[float, float],
    kernel_size: int,
    merge_radius: float,
    matching_radius: float,
    use_tta: bool,
    output_hw: tuple[int, int],
    checkpoint_name: str,
) -> tuple[list[dict[str, str | int | float]], dict[str, float | int]]:
    rows: list[dict[str, str | int | float]] = []

    tp_pos = fp_pos = fn_pos = 0
    tp_neg = fp_neg = fn_neg = 0

    model.eval()
    with torch.no_grad():
        for sample_idx, (img_batch, _, pos_pts_tuple, neg_pts_tuple) in enumerate(loader):
            img_batch = img_batch.to(device, non_blocking=True)
            pred_heatmaps = tta_forward(model, img_batch) if use_tta else torch.sigmoid(model(img_batch))

            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=pred_heatmaps,
                kernel_size=kernel_size,
                threshold=threshold,
                merge_radius=merge_radius,
                output_hw=output_hw,
            )

            pred_pos = out_pos[0]
            pred_neg = out_neg[0]

            gt_pos = pos_pts_tuple[0].astype(np.float32)
            gt_neg = neg_pts_tuple[0].astype(np.float32)

            tp, fp, fn = compute_metrics(pred_pos, gt_pos, radius=matching_radius)
            tp_pos += tp
            fp_pos += fp
            fn_pos += fn

            tp, fp, fn = compute_metrics(pred_neg, gt_neg, radius=matching_radius)
            tp_neg += tp
            fp_neg += fp
            fn_neg += fn

            image_path = dataset.samples[sample_idx][0]
            image_id = image_path.stem
            rel_image_path = image_path.relative_to(dataset.root).as_posix()
            heatmap_pos = pred_heatmaps[0, 0].detach().cpu()
            heatmap_neg = pred_heatmaps[0, 1].detach().cpu()

            rows.extend(
                build_raw_prediction_rows(
                    fold=fold_index,
                    image_id=image_id,
                    image_path=rel_image_path,
                    points_xy=pred_pos,
                    pred_class="pos",
                    heatmap_pos=heatmap_pos,
                    heatmap_neg=heatmap_neg,
                    output_hw=output_hw,
                    threshold=threshold,
                    model_checkpoint=checkpoint_name,
                )
            )
            rows.extend(
                build_raw_prediction_rows(
                    fold=fold_index,
                    image_id=image_id,
                    image_path=rel_image_path,
                    points_xy=pred_neg,
                    pred_class="neg",
                    heatmap_pos=heatmap_pos,
                    heatmap_neg=heatmap_neg,
                    output_hw=output_hw,
                    threshold=threshold,
                    model_checkpoint=checkpoint_name,
                )
            )

    precision_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
    recall_pos = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0.0
    f1_pos = (
        2.0 * precision_pos * recall_pos / (precision_pos + recall_pos)
        if (precision_pos + recall_pos) > 0
        else 0.0
    )

    precision_neg = tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0.0
    recall_neg = tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else 0.0
    f1_neg = (
        2.0 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0
        else 0.0
    )

    tp_both = tp_pos + tp_neg
    fp_both = fp_pos + fp_neg
    fn_both = fn_pos + fn_neg

    precision = tp_both / (tp_both + fp_both) if (tp_both + fp_both) > 0 else 0.0
    recall = tp_both / (tp_both + fn_both) if (tp_both + fn_both) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    f1_macro = 0.5 * (f1_pos + f1_neg)

    metrics = {
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
        "tp_pos": tp_pos,
        "fp_pos": fp_pos,
        "fn_pos": fn_pos,
        "tp_neg": tp_neg,
        "fp_neg": fp_neg,
        "fn_neg": fn_neg,
        "num_prediction_rows": len(rows),
    }
    return rows, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one fold on BCData train images and export raw OOF holdout predictions."
    )
    parser.add_argument("--config", default="configs/train_mse.yaml")
    parser.add_argument("--fold-dir", required=True, type=Path)
    parser.add_argument("--fold-index", required=True, type=int)
    parser.add_argument("--out-dir", default="oof")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device-key", default="h200", choices=["hpvictus", "collab", "h200"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_manifest = args.fold_dir / f"fold_{args.fold_index}_train_images.txt"
    holdout_manifest = args.fold_dir / f"fold_{args.fold_index}_holdout_images.txt"
    if not train_manifest.exists():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest}")
    if not holdout_manifest.exists():
        raise FileNotFoundError(f"Missing holdout manifest: {holdout_manifest}")

    train_image_ids = load_image_ids(train_manifest)
    holdout_image_ids = load_image_ids(holdout_manifest)

    if train_image_ids & holdout_image_ids:
        raise ValueError("Train and holdout manifests overlap, which would cause leakage.")

    set_seed(int(cfg["seed"]))

    root_dir, checkpoint_dir = get_paths(device=args.device_key)
    run_dir = build_run_dir(
        checkpoint_dir=checkpoint_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        fold_index=args.fold_index,
    )
    print(f"RUN_DIR:{run_dir}")

    shutil.copy(args.config, run_dir / "config.yaml")
    shutil.copy(train_manifest, run_dir / train_manifest.name)
    shutil.copy(holdout_manifest, run_dir / holdout_manifest.name)

    threshold = resolve_threshold(cfg)
    print(f"Validation thresholds: thr_pos={threshold[0]:.3f}, thr_neg={threshold[1]:.3f}")

    heatmap_gen = LocHeatmap(
        out_hw=tuple(cfg["out_hw"]),
        in_hw=tuple(cfg["in_hw"]),
        sigma=cfg["sigma"],
        dtype=torch.float32,
    )

    joint_tf = build_train_transform() if cfg.get("augmentation", False) else None
    persistent_workers = int(cfg["num_workers"]) > 0

    dataset_train = BCDataDataset(
        root=root_dir,
        split="train",
        target_transform=heatmap_gen,
        joint_transform=joint_tf,
        input_normalization=cfg["input_normalization"],
        image_ids=train_image_ids,
    )
    dataset_val = BCDataDataset(
        root=root_dir,
        split="validation",
        target_transform=heatmap_gen,
        input_normalization=cfg["input_normalization"],
    )
    dataset_holdout = BCDataDataset(
        root=root_dir,
        split="train",
        target_transform=heatmap_gen,
        input_normalization=cfg["input_normalization"],
        image_ids=holdout_image_ids,
    )

    expected_train_images = len(train_image_ids)
    expected_holdout_images = len(holdout_image_ids)
    if len(dataset_train) != expected_train_images:
        raise RuntimeError(
            f"Train subset size mismatch: dataset={len(dataset_train)} manifest={expected_train_images}"
        )
    if len(dataset_holdout) != expected_holdout_images:
        raise RuntimeError(
            f"Holdout subset size mismatch: dataset={len(dataset_holdout)} manifest={expected_holdout_images}"
        )

    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))

    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
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
    dataloader_holdout = DataLoader(
        dataset=dataset_holdout,
        batch_size=1,
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
    model = KiLocNet(pretrained=cfg["is_pretrained"], backbone_name=cfg["backbone"]).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = build_scheduler(cfg, optimizer)

    use_ema = cfg.get("ema", False)
    ema = None
    ema_start_epoch = cfg.get("ema_start_epoch", 5)
    best_f1_macro = -1.0
    best_epoch = -1
    best_eval_checkpoint_path: Path | None = None
    history: list[dict[str, float | int]] = []

    for epoch_idx in range(int(cfg["epochs"])):
        if use_ema and ema is None and epoch_idx == ema_start_epoch:
            ema = ModelEMA(
                model=model,
                decay=cfg.get("ema_decay", 0.999),
                device=cfg.get("ema_device", None),
            )
            print(f"EMA initialized at epoch {epoch_idx + 1}")

        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            trainloader=dataloader_train,
            ema=ema,
        )

        eval_model = ema.module if ema is not None else model
        val_result = val_one_epoch(
            model=eval_model,
            criterion=criterion,
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
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"P={precision:.3f}, R={recall:.3f}, F1_micro={f1:.3f}, F1_macro={f1_macro:.3f}"
        )

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_epoch = epoch_idx + 1
            best_eval_checkpoint_path = save_single_best_checkpoint(
                run_dir=run_dir,
                best_epoch=best_epoch,
                model=model,
                ema_module=ema.module if ema is not None else None,
                previous_checkpoint_path=best_eval_checkpoint_path,
            )

        history.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": train_loss,
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
        )
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if scheduler is not None:
            scheduler.step(f1_macro)

    if best_eval_checkpoint_path is None:
        raise RuntimeError("Training finished without saving a best checkpoint.")

    inference_model = KiLocNet(
        pretrained=False,
        backbone_name=cfg["backbone"],
    )
    inference_model.load_state_dict(torch.load(best_eval_checkpoint_path, map_location="cpu"))
    inference_model = inference_model.to(device)

    raw_rows, holdout_metrics = infer_holdout(
        model=inference_model,
        loader=dataloader_holdout,
        dataset=dataset_holdout,
        device=device,
        fold_index=args.fold_index,
        threshold=threshold,
        kernel_size=cfg["kernel_size"],
        merge_radius=cfg["merge_radius"],
        matching_radius=cfg["matching_radius"],
        use_tta=cfg.get("tta", False),
        output_hw=tuple(cfg["in_hw"]),
        checkpoint_name=best_eval_checkpoint_path.name,
    )

    raw_csv_path = run_dir / f"fold_{args.fold_index}_raw_predictions.csv"
    write_raw_prediction_csv(raw_rows, raw_csv_path)

    holdout_image_paths = [image_path.relative_to(root_dir).as_posix() for image_path, _, _ in dataset_holdout.samples]
    gt_by_image = load_gt_by_image_from_image_paths(root_dir, holdout_image_paths)
    relation_rows = build_relation_rows(
        raw_rows=raw_rows,
        gt_by_image=gt_by_image,
        tau_mine=threshold,
        matching_radius=float(cfg["matching_radius"]),
        r_exclude_any=1.5 * float(cfg["matching_radius"]),
        r_cluster_same=0.75 * float(cfg["matching_radius"]),
        r_interclass_conflict=float(cfg["matching_radius"]),
    )
    relation_summary = summarize_relation_rows(
        relation_rows,
        total_holdout_images=len(holdout_image_paths),
    )
    relation_summary.update(
        {
            "fold_index": args.fold_index,
            "matching_radius": float(cfg["matching_radius"]),
            "tau_mine_pos": threshold[0],
            "tau_mine_neg": threshold[1],
            "r_exclude_any": 1.5 * float(cfg["matching_radius"]),
            "r_cluster_same": 0.75 * float(cfg["matching_radius"]),
            "r_interclass_conflict": float(cfg["matching_radius"]),
        }
    )
    relations_path, relation_summary_path = relation_artifact_paths(
        run_dir,
        fold_index=args.fold_index,
    )
    write_relation_csv(relation_rows, relations_path)
    write_relation_summary(relation_summary, relation_summary_path)

    metadata = {
        "fold_index": args.fold_index,
        "train_manifest": train_manifest.name,
        "holdout_manifest": holdout_manifest.name,
        "num_train_images": len(train_image_ids),
        "num_holdout_images": len(holdout_image_ids),
        "num_train_plus_holdout_images": len(train_image_ids) + len(holdout_image_ids),
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1_macro,
        "best_checkpoint": best_eval_checkpoint_path.name,
        "best_eval_checkpoint": best_eval_checkpoint_path.name,
        "threshold_pos": threshold[0],
        "threshold_neg": threshold[1],
        "relation_csv": relations_path.relative_to(run_dir).as_posix(),
        "relation_summary_json": relation_summary_path.relative_to(run_dir).as_posix(),
    }

    with (run_dir / "oof_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (run_dir / f"fold_{args.fold_index}_holdout_results.json").open("w", encoding="utf-8") as f:
        json.dump(holdout_metrics, f, indent=2)

    print(f"BEST EPOCH WAS: {best_epoch}")
    print(f"Saved raw holdout predictions to: {raw_csv_path}")
    print(f"Saved prediction relations to: {relations_path}")


if __name__ == "__main__":
    main()
