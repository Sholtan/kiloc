"""Validate a trained point classifier checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from kiloc.datasets import BCDataPointDataset
from kiloc.model.classifier_net import ResNetClassifier
from kiloc.training.train_classifier import validate_one_epoch
from kiloc.utils.config import get_paths

from run_train import read_background_rgb, set_seed, seed_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to training run directory (must contain config.yaml and a .pth checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename inside run_dir. If omitted, uses the *best* .pth file found.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the threshold from config.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep thresholds 0.05–0.95 and report the best f1_macro.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    # ── Find checkpoint ──
    if args.checkpoint:
        ckpt_path = run_dir / args.checkpoint
    else:
        # Pick the best checkpoint (by filename convention)
        pth_files = sorted(run_dir.glob("*best*.pth"))
        if not pth_files:
            pth_files = sorted(run_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {run_dir}")
        ckpt_path = pth_files[0]

    print(f"Checkpoint: {ckpt_path.name}")

    # ── Build dataset ──
    root_dir, _ = get_paths(device=cfg.get("device_key", "h200"))

    background_rgb = read_background_rgb(
        background_stats_path=cfg["background_stats_path"],
    )

    if args.split == "val":
        points_csv = Path(cfg.get("val_points_csv", "bcdata_point_tables/validation_points.csv"))
    else:
        points_csv = Path(cfg.get("train_points_csv", "bcdata_point_tables/train_points.csv"))

    dataset = BCDataPointDataset(
        data_root=root_dir,
        points_csv=points_csv,
        crop_size=cfg["crop_size"],
        resize_to=cfg["input_size"],
        background_rgb=background_rgb,
        image_transform=None,
        input_normalization=cfg.get("input_normalization", "imagenet"),
    )

    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    num_workers = cfg.get("num_workers", 4)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 128),
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=BCDataPointDataset.collate_fn,
        pin_memory=True,
        drop_last=False,
        generator=g,
        persistent_workers=num_workers > 0,
    )

    # ── Build model ──
    model = ResNetClassifier(
        backbone_name=cfg["backbone"],
        input_size=cfg["input_size"],
        num_classes=1,
        pretrained=False,
        dropout=cfg.get("dropout", 0.0),
        adapt_small_inputs=cfg.get("adapt_small_inputs", True),
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    use_amp = cfg.get("use_amp", True)

    # ── Single threshold evaluation ──
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.5)

    val_loss, metrics = validate_one_epoch(
        model=model,
        criterion=criterion,
        device=device,
        val_loader=loader,
        threshold=threshold,
        use_amp=use_amp,
    )

    print(f"\n── {args.split} split @ threshold={threshold:.3f} ──")
    print(f"Loss:              {val_loss:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 macro:          {metrics['f1_macro']:.4f}")
    print(f"  Pos  P={metrics['precision_pos']:.3f}  R={metrics['recall_pos']:.3f}  F1={metrics['f1_pos']:.3f}")
    print(f"  Neg  P={metrics['precision_neg']:.3f}  R={metrics['recall_neg']:.3f}  F1={metrics['f1_neg']:.3f}")
    print(f"  TP={int(metrics['tp'])}  FP={int(metrics['fp'])}  FN={int(metrics['fn'])}  TN={int(metrics['tn'])}")

    # ── Threshold sweep ──
    if args.sweep:
        print("\n── Threshold sweep ──")
        thresholds = np.arange(0.05, 0.96, 0.05)
        best_thr = threshold
        best_f1 = metrics["f1_macro"]
        results = []

        for thr in thresholds:
            _, m = validate_one_epoch(
                model=model,
                criterion=criterion,
                device=device,
                val_loader=loader,
                threshold=float(thr),
                use_amp=use_amp,
            )
            results.append({"threshold": round(float(thr), 3), **m})
            tag = " <-- best" if m["f1_macro"] > best_f1 else ""
            print(f"  thr={thr:.3f}  f1_macro={m['f1_macro']:.4f}  bal_acc={m['balanced_accuracy']:.4f}{tag}")
            if m["f1_macro"] > best_f1:
                best_f1 = m["f1_macro"]
                best_thr = float(thr)

        print(f"\nBest threshold: {best_thr:.3f}  (f1_macro={best_f1:.4f})")

        sweep_path = run_dir / f"threshold_sweep_{args.split}.json"
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump({"best_threshold": best_thr, "best_f1_macro": best_f1, "results": results}, f, indent=2)
        print(f"Saved to {sweep_path}")


if __name__ == "__main__":
    main()