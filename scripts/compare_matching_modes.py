from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.losses.losses import SigmoidSumHuber, SigmoidWeightedMSE, sigmoid_focal_loss
from kiloc.model.kiloc_net import KiLocNet
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.training.train import val_one_epoch_detailed
from kiloc.utils.config import get_paths


BEST_CKPT_RE = re.compile(r"^kilocnet_best_f1_epoch_(\d+)(?:_ema)?\.pth$")
RUN_DIR_RE = re.compile(r"^(backbone_[^_]+_seed_\d+)$")


def build_criterion(cfg: dict[str, Any]):
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


def resolve_threshold(cfg: dict[str, Any]) -> tuple[float, float]:
    base_threshold = float(cfg.get("threshold", 0.5))
    return (
        float(cfg.get("thr_pos", base_threshold)),
        float(cfg.get("thr_neg", base_threshold)),
    )


def select_best_checkpoint(run_dir: Path) -> Path:
    best_ema = sorted(run_dir.glob("kilocnet_best_f1_epoch_*_ema.pth"))
    if best_ema:
        return best_ema[0]

    best_plain = [
        path
        for path in sorted(run_dir.glob("kilocnet_best_f1_epoch_*.pth"))
        if not path.name.endswith("_ema.pth")
    ]
    if best_plain:
        return best_plain[0]

    raise FileNotFoundError(f"Could not find best checkpoint under {run_dir}")


def load_reference_metrics(run_dir: Path, grid_results_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.json in {run_dir}")

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    grid_entry = grid_results_map.get(run_dir.name)
    if grid_entry is not None:
        best_epoch = int(grid_entry["best_epoch"])
    else:
        best_epoch = int(max(history, key=lambda row: row["f1_macro"])["epoch"])

    history_entry = next((row for row in history if int(row["epoch"]) == best_epoch), None)
    if history_entry is None:
        raise RuntimeError(f"Could not find epoch {best_epoch} in {history_path}")

    validation_results_path = run_dir / "validation_results.json"
    validation_results = None
    if validation_results_path.exists():
        with validation_results_path.open("r", encoding="utf-8") as f:
            validation_results = json.load(f)

    return {
        "best_epoch": best_epoch,
        "history": history_entry,
        "grid_results": grid_entry,
        "validation_results": validation_results,
    }


def compare_metrics(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    keys: list[str],
) -> dict[str, float]:
    return {
        key: float(actual[key]) - float(expected[key])
        for key in keys
    }


def evaluate_run(
    run_dir: Path,
    *,
    split: str,
    device_key: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
    matching_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_dir, _ = get_paths(device=device_key)
    checkpoint_path = select_best_checkpoint(run_dir)

    heatmap_gen = LocHeatmap(
        out_hw=tuple(cfg["out_hw"]),
        in_hw=tuple(cfg["in_hw"]),
        sigma=cfg["sigma"],
        dtype=torch.float32,
    )
    dataset = BCDataDataset(
        root=root_dir,
        split=split,
        target_transform=heatmap_gen,
        input_normalization=cfg["input_normalization"],
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size_override if batch_size_override is not None else int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=(
            num_workers_override
            if num_workers_override is not None
            else int(cfg.get("num_workers", 4))
        ),
        pin_memory=True,
        drop_last=False,
    )

    model = KiLocNet(pretrained=False, backbone_name=cfg["backbone"])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    metrics = val_one_epoch_detailed(
        model=model,
        criterion=build_criterion(cfg),
        device=device,
        val_loader=loader,
        kernel_size=int(cfg["kernel_size"]),
        threshold=resolve_threshold(cfg),
        merge_radius=float(cfg["merge_radius"]),
        matching_radius=float(cfg["matching_radius"]),
        tta=bool(cfg.get("tta", False)),
        matching_mode=matching_mode,
    )

    eval_context = {
        "checkpoint": checkpoint_path.name,
        "threshold_pos": resolve_threshold(cfg)[0],
        "threshold_neg": resolve_threshold(cfg)[1],
        "merge_radius": float(cfg["merge_radius"]),
        "matching_radius": float(cfg["matching_radius"]),
        "tta": bool(cfg.get("tta", False)),
        "matching_mode": matching_mode,
    }
    return metrics, eval_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare greedy vs optimal validation matching for a grid run."
    )
    parser.add_argument("--gridrun-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--device-key", default="h200", choices=["hpvictus", "collab", "h200"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--matching-mode-old", default="greedy", choices=["greedy", "optimal"])
    parser.add_argument("--matching-mode-new", default="optimal", choices=["greedy", "optimal"])
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gridrun_dir = args.gridrun_dir

    grid_results_path = gridrun_dir / "grid_results.json"
    grid_results_map: dict[str, dict[str, Any]] = {}
    if grid_results_path.exists():
        with grid_results_path.open("r", encoding="utf-8") as f:
            grid_results = json.load(f)
        for entry in grid_results:
            run_name = Path(entry["run_dir"]).name
            grid_results_map[run_name] = entry

    run_dirs = sorted(
        path for path in gridrun_dir.iterdir()
        if path.is_dir() and RUN_DIR_RE.match(path.name)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {gridrun_dir}")

    metric_keys = [
        "precision",
        "recall",
        "f1",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "precision_neg",
        "recall_neg",
        "f1_neg",
        "f1_macro",
    ]

    results: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        reference = load_reference_metrics(run_dir, grid_results_map)
        old_metrics, eval_context = evaluate_run(
            run_dir,
            split=args.split,
            device_key=args.device_key,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            matching_mode=args.matching_mode_old,
        )
        new_metrics, _ = evaluate_run(
            run_dir,
            split=args.split,
            device_key=args.device_key,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            matching_mode=args.matching_mode_new,
        )

        history_deltas = compare_metrics(reference["history"], old_metrics, keys=metric_keys)
        validation_deltas = None
        if reference["validation_results"] is not None:
            validation_deltas = compare_metrics(
                reference["validation_results"],
                old_metrics,
                keys=metric_keys,
            )

        result = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "reference_best_epoch": reference["best_epoch"],
            "reference_history_metrics": {key: reference["history"][key] for key in metric_keys},
            "reference_grid_metrics": (
                {key: reference["grid_results"][key] for key in ["precision", "recall", "f1_macro"]}
                if reference["grid_results"] is not None
                else None
            ),
            "reference_validation_results": reference["validation_results"],
            "evaluation_context": eval_context,
            "recomputed_old_metrics": old_metrics,
            "recomputed_new_metrics": new_metrics,
            "old_minus_reference_history": history_deltas,
            "old_matches_reference_history": all(abs(delta) <= 1e-6 for delta in history_deltas.values()),
            "old_minus_validation_results": validation_deltas,
            "old_matches_validation_results": (
                None
                if validation_deltas is None
                else all(abs(delta) <= 1e-6 for delta in validation_deltas.values())
            ),
            "new_minus_old": compare_metrics(old_metrics, new_metrics, keys=metric_keys),
            "new_count_deltas": {
                "tp_pos": int(new_metrics["tp_pos"]) - int(old_metrics["tp_pos"]),
                "fp_pos": int(new_metrics["fp_pos"]) - int(old_metrics["fp_pos"]),
                "fn_pos": int(new_metrics["fn_pos"]) - int(old_metrics["fn_pos"]),
                "tp_neg": int(new_metrics["tp_neg"]) - int(old_metrics["tp_neg"]),
                "fp_neg": int(new_metrics["fp_neg"]) - int(old_metrics["fp_neg"]),
                "fn_neg": int(new_metrics["fn_neg"]) - int(old_metrics["fn_neg"]),
            },
        }
        results.append(result)

        print(
            f"{run_dir.name} | "
            f"old_f1_macro={old_metrics['f1_macro']:.6f} "
            f"(matches_history={result['old_matches_reference_history']}) | "
            f"new_f1_macro={new_metrics['f1_macro']:.6f} | "
            f"delta={result['new_minus_old']['f1_macro']:+.6f}"
        )

    out_path = (
        args.out_json
        if args.out_json is not None
        else gridrun_dir / f"{args.split}_matching_comparison.json"
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
