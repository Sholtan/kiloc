from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.evaluation.metrics import compute_metrics, match_indices
from kiloc.evaluation.tta import tta_forward
from kiloc.model.kiloc_net import KiLocNet
from kiloc.oof.predictions import build_raw_prediction_rows
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.three_classifier.datasets import (
    crop_uint8_to_tensor,
    extract_center_crop_uint8,
    read_image_rgb,
)
from kiloc.three_classifier.model import ThreeClassCropClassifier
from kiloc.utils.config import get_paths


def build_default_tau_grid() -> list[float]:
    return [float(x) for x in np.linspace(0.0, 1.0, 101)]


def resolve_output_dir(
    *,
    localization_run_dir: str | Path,
    classifier_run_dir: str | Path,
    out_dir: str | Path | None,
) -> Path:
    if out_dir is not None:
        resolved = Path(out_dir)
    else:
        resolved = (
            Path(localization_run_dir)
            / "three_classifier_filter_eval"
            / Path(classifier_run_dir).name
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_localization_checkpoint(
    run_dir: str | Path,
    checkpoint: str | Path | None = None,
) -> Path:
    return _resolve_checkpoint(
        run_dir=run_dir,
        explicit_checkpoint=checkpoint,
        patterns=(
            "kilocnet_best_f1_epoch_*_ema.pth",
            "kilocnet_best_f1_epoch_*.pth",
            "*best*_ema.pth",
            "*best*.pth",
            "*.pth",
        ),
    )


def resolve_classifier_checkpoint(
    run_dir: str | Path,
    checkpoint: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    if checkpoint is None:
        run_summary_path = run_dir / "run_summary.json"
        if run_summary_path.exists():
            with run_summary_path.open("r", encoding="utf-8") as handle:
                run_summary = json.load(handle)
            best_checkpoint = run_summary.get("best_checkpoint")
            if isinstance(best_checkpoint, str) and best_checkpoint:
                candidate = run_dir / best_checkpoint
                if candidate.exists():
                    return candidate

    return _resolve_checkpoint(
        run_dir=run_dir,
        explicit_checkpoint=checkpoint,
        patterns=(
            "three_classifier_best_f1_epoch_*.pth",
            "*best*.pth",
            "*.pth",
        ),
    )


def _resolve_checkpoint(
    *,
    run_dir: str | Path,
    explicit_checkpoint: str | Path | None,
    patterns: tuple[str, ...],
) -> Path:
    run_dir = Path(run_dir)
    if explicit_checkpoint is not None:
        checkpoint_path = Path(explicit_checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        candidate = run_dir / checkpoint_path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find checkpoint {explicit_checkpoint!r}")

    for pattern in patterns:
        matches = sorted(path for path in run_dir.glob(pattern) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[-1]

    raise FileNotFoundError(f"Could not resolve a checkpoint under {run_dir}")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_validation_dataset(
    *,
    data_root: str | Path,
    localization_cfg: dict[str, Any],
    split: str,
) -> BCDataDataset:
    heatmap_gen = LocHeatmap(
        out_hw=tuple(localization_cfg["out_hw"]),
        in_hw=tuple(localization_cfg["in_hw"]),
        sigma=float(localization_cfg["sigma"]),
        dtype=torch.float32,
    )
    return BCDataDataset(
        root=Path(data_root),
        split=split,
        target_transform=heatmap_gen,
        input_normalization=localization_cfg.get("input_normalization"),
    )


def _load_localization_model(
    *,
    localization_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> KiLocNet:
    model = KiLocNet(
        pretrained=False,
        backbone_name=localization_cfg["backbone"],
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _load_classifier_model(
    *,
    classifier_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> ThreeClassCropClassifier:
    model = ThreeClassCropClassifier(
        pretrained=False,
        embedding_dim=int(classifier_cfg.get("embedding_dim", 128)),
        num_classes=3,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _build_classifier_probabilities(
    *,
    classifier_model: ThreeClassCropClassifier,
    image_rgb: np.ndarray,
    rows: list[dict[str, Any]],
    crop_size: int,
    resize_hw: tuple[int, int],
    input_normalization: str | None,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, float | int]]:
    if not rows:
        return []

    tensors: list[torch.Tensor] = []
    for row in rows:
        crop_rgb = extract_center_crop_uint8(
            image_rgb,
            center_x=float(row["x"]),
            center_y=float(row["y"]),
            crop_size=crop_size,
            resize_hw=resize_hw,
        )
        tensors.append(
            crop_uint8_to_tensor(
                crop_rgb,
                input_normalization=input_normalization,
            )
        )

    outputs: list[dict[str, float | int]] = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size], dim=0).to(device, non_blocking=True)
            probs = torch.softmax(classifier_model(batch), dim=1).cpu().numpy()
            for prob in probs:
                pred_label = int(prob.argmax())
                outputs.append(
                    {
                        "classifier_pred_label": pred_label,
                        "classifier_prob_class_0": float(prob[0]),
                        "classifier_prob_class_1": float(prob[1]),
                        "classifier_prob_class_2": float(prob[2]),
                        "p_mined": float(prob[2]),
                    }
                )
    return outputs


def _baseline_match_metadata(
    *,
    image_rows: list[dict[str, Any]],
    gt_pos: np.ndarray,
    gt_neg: np.ndarray,
    matching_radius: float,
    matching_mode: str,
) -> None:
    for row in image_rows:
        row["baseline_match_status"] = "unmatched"
        row["is_baseline_matched"] = 0
        row["baseline_matched_gt_class"] = ""
        row["baseline_matched_gt_id"] = ""
        row["baseline_matched_dist"] = ""

    for pred_class, gt_points in (("pos", gt_pos), ("neg", gt_neg)):
        class_rows = [row for row in image_rows if row["pred_class"] == pred_class]
        if not class_rows:
            continue
        pred_points = np.asarray([[row["x"], row["y"]] for row in class_rows], dtype=np.float32)
        matched_pred, matched_gt = match_indices(
            pred_pts=pred_points,
            gt_pts=gt_points.astype(np.float32),
            radius=matching_radius,
            matching_mode=matching_mode,
        )
        for pred_idx, gt_idx in zip(matched_pred.tolist(), matched_gt.tolist()):
            row = class_rows[int(pred_idx)]
            row["baseline_match_status"] = "matched"
            row["is_baseline_matched"] = 1
            row["baseline_matched_gt_class"] = pred_class
            row["baseline_matched_gt_id"] = f"{pred_class}_{int(gt_idx):04d}"
            dist = float(np.linalg.norm(pred_points[int(pred_idx)] - gt_points[int(gt_idx)].astype(np.float32)))
            row["baseline_matched_dist"] = dist


def _should_keep(*, p_mined: float, tau: float) -> bool:
    return bool(tau >= 1.0 or p_mined < tau)


def _build_metrics_dict(
    *,
    tp_pos: int,
    fp_pos: int,
    fn_pos: int,
    tp_neg: int,
    fp_neg: int,
    fn_neg: int,
) -> dict[str, float | int]:
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

    tp_total = tp_pos + tp_neg
    fp_total = fp_pos + fp_neg
    fn_total = fn_pos + fn_neg
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp_pos": tp_pos,
        "fp_pos": fp_pos,
        "fn_pos": fn_pos,
        "tp_neg": tp_neg,
        "fp_neg": fp_neg,
        "fn_neg": fn_neg,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "f1_macro": 0.5 * (f1_pos + f1_neg),
    }


def _evaluate_tau(
    *,
    proposal_rows: list[dict[str, Any]],
    gt_by_image: dict[str, dict[str, Any]],
    tau: float,
    matching_radius: float,
    matching_mode: str,
) -> dict[str, float | int]:
    rows_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in proposal_rows:
        rows_by_image.setdefault(str(row["image_id"]), []).append(row)

    tp_pos = fp_pos = fn_pos = 0
    tp_neg = fp_neg = fn_neg = 0
    kept_count = rejected_count = 0
    matched_kept = matched_rejected = 0
    unmatched_kept = unmatched_rejected = 0

    for image_id, gt in gt_by_image.items():
        image_rows = rows_by_image.get(image_id, [])
        kept_rows = [row for row in image_rows if _should_keep(p_mined=float(row["p_mined"]), tau=tau)]
        kept_pred_ids = {str(row["detector_pred_id"]) for row in kept_rows}
        rejected_rows = [
            row for row in image_rows
            if str(row["detector_pred_id"]) not in kept_pred_ids
        ]

        kept_count += len(kept_rows)
        rejected_count += len(rejected_rows)
        matched_kept += sum(int(row["is_baseline_matched"]) for row in kept_rows)
        matched_rejected += sum(int(row["is_baseline_matched"]) for row in rejected_rows)
        unmatched_kept += sum(1 - int(row["is_baseline_matched"]) for row in kept_rows)
        unmatched_rejected += sum(1 - int(row["is_baseline_matched"]) for row in rejected_rows)

        pos_pred = np.asarray(
            [[row["x"], row["y"]] for row in kept_rows if row["detector_pred_class"] == "pos"],
            dtype=np.float32,
        ).reshape(-1, 2)
        neg_pred = np.asarray(
            [[row["x"], row["y"]] for row in kept_rows if row["detector_pred_class"] == "neg"],
            dtype=np.float32,
        ).reshape(-1, 2)

        tp, fp, fn = compute_metrics(
            pred_pts=pos_pred,
            gt_pts=np.asarray(gt["pos_points"], dtype=np.float32),
            radius=matching_radius,
            matching_mode=matching_mode,
        )
        tp_pos += tp
        fp_pos += fp
        fn_pos += fn

        tp, fp, fn = compute_metrics(
            pred_pts=neg_pred,
            gt_pts=np.asarray(gt["neg_points"], dtype=np.float32),
            radius=matching_radius,
            matching_mode=matching_mode,
        )
        tp_neg += tp
        fp_neg += fp
        fn_neg += fn

    metrics = _build_metrics_dict(
        tp_pos=tp_pos,
        fp_pos=fp_pos,
        fn_pos=fn_pos,
        tp_neg=tp_neg,
        fp_neg=fp_neg,
        fn_neg=fn_neg,
    )
    metrics.update(
        {
            "tau": float(tau),
            "total_proposals": int(len(proposal_rows)),
            "kept_count": int(kept_count),
            "rejected_count": int(rejected_count),
            "matched_kept": int(matched_kept),
            "matched_rejected": int(matched_rejected),
            "unmatched_kept": int(unmatched_kept),
            "unmatched_rejected": int(unmatched_rejected),
        }
    )
    return metrics


def _collect_validation_proposals(
    *,
    localization_model: KiLocNet,
    classifier_model: ThreeClassCropClassifier,
    dataset: BCDataDataset,
    loader: DataLoader,
    localization_cfg: dict[str, Any],
    classifier_cfg: dict[str, Any],
    localization_checkpoint_name: str,
    classifier_checkpoint_name: str,
    split: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    threshold = (
        float(localization_cfg.get("thr_pos", localization_cfg.get("threshold", 0.5))),
        float(localization_cfg.get("thr_neg", localization_cfg.get("threshold", 0.5))),
    )
    kernel_size = int(localization_cfg["kernel_size"])
    merge_radius = float(localization_cfg["merge_radius"])
    matching_radius = float(localization_cfg["matching_radius"])
    output_hw = tuple(int(x) for x in localization_cfg.get("in_hw", [640, 640]))
    matching_mode = str(localization_cfg.get("matching_mode", "optimal"))

    classifier_dataset_cfg = classifier_cfg["dataset"]
    crop_size = int(classifier_dataset_cfg["crop_size"])
    resize_hw = tuple(int(x) for x in classifier_dataset_cfg.get("resize_hw", [128, 128]))
    classifier_input_normalization = classifier_dataset_cfg.get("input_normalization", "imagenet")
    classifier_batch_size = int(classifier_cfg.get("batch_size", 128))

    image_cache: dict[str, np.ndarray] = {}
    proposal_rows: list[dict[str, Any]] = []
    gt_by_image: dict[str, dict[str, Any]] = {}

    localization_model.eval()
    classifier_model.eval()

    with torch.no_grad():
        for sample_idx, (img_batch, _heatmaps_batch, pos_pts_tuple, neg_pts_tuple) in enumerate(
            tqdm(loader, desc=f"{split} proposals", leave=False)
        ):
            img_batch = img_batch.to(device, non_blocking=True)
            pred_heatmaps = (
                tta_forward(localization_model, img_batch)
                if localization_cfg.get("tta", False)
                else torch.sigmoid(localization_model(img_batch))
            )

            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=pred_heatmaps,
                kernel_size=kernel_size,
                threshold=threshold,
                merge_radius=merge_radius,
                output_hw=output_hw,
            )

            image_path = dataset.samples[sample_idx][0]
            rel_image_path = image_path.relative_to(dataset.root).as_posix()
            image_id = image_path.stem

            heatmap_pos = pred_heatmaps[0, 0].detach().cpu()
            heatmap_neg = pred_heatmaps[0, 1].detach().cpu()
            image_rows = []
            image_rows.extend(
                build_raw_prediction_rows(
                    fold=-1,
                    image_id=image_id,
                    image_path=rel_image_path,
                    points_xy=out_pos[0],
                    pred_class="pos",
                    heatmap_pos=heatmap_pos,
                    heatmap_neg=heatmap_neg,
                    output_hw=output_hw,
                    threshold=threshold,
                    model_checkpoint=localization_checkpoint_name,
                )
            )
            image_rows.extend(
                build_raw_prediction_rows(
                    fold=-1,
                    image_id=image_id,
                    image_path=rel_image_path,
                    points_xy=out_neg[0],
                    pred_class="neg",
                    heatmap_pos=heatmap_pos,
                    heatmap_neg=heatmap_neg,
                    output_hw=output_hw,
                    threshold=threshold,
                    model_checkpoint=localization_checkpoint_name,
                )
            )

            gt_pos = pos_pts_tuple[0].astype(np.float32)
            gt_neg = neg_pts_tuple[0].astype(np.float32)
            gt_by_image[image_id] = {
                "image_id": image_id,
                "image_path": rel_image_path,
                "split": split,
                "pos_points": gt_pos,
                "neg_points": gt_neg,
            }

            if not image_rows:
                continue

            _baseline_match_metadata(
                image_rows=image_rows,
                gt_pos=gt_pos,
                gt_neg=gt_neg,
                matching_radius=matching_radius,
                matching_mode=matching_mode,
            )

            image_rgb = read_image_rgb(image_path, cache=image_cache)
            classifier_outputs = _build_classifier_probabilities(
                classifier_model=classifier_model,
                image_rgb=image_rgb,
                rows=image_rows,
                crop_size=crop_size,
                resize_hw=resize_hw,
                input_normalization=classifier_input_normalization,
                device=device,
                batch_size=classifier_batch_size,
            )

            for row, classifier_output in zip(image_rows, classifier_outputs):
                row.update(classifier_output)
                row["split"] = split
                row["detector_pred_class"] = row.pop("pred_class")
                row["detector_pred_id"] = row.pop("pred_id")
                row["detector_score"] = row.pop("score")
                row["detector_score_pos"] = row.pop("score_pos")
                row["detector_score_neg"] = row.pop("score_neg")
                row["localization_checkpoint"] = localization_checkpoint_name
                row["classifier_checkpoint"] = classifier_checkpoint_name
                row["classifier_crop_size"] = crop_size
                row["classifier_resize_h"] = resize_hw[0]
                row["classifier_resize_w"] = resize_hw[1]
                row["classifier_input_normalization"] = classifier_input_normalization or ""
                row["kept_at_best_tau"] = ""
                row["rejected_at_best_tau"] = ""

            proposal_rows.extend(image_rows)

    proposal_rows.sort(
        key=lambda row: (
            row["image_path"],
            row["detector_pred_class"],
            int(row["raw_rank_in_image"]),
        )
    )
    return proposal_rows, gt_by_image


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_filter_on_localization(
    *,
    localization_run_dir: str | Path,
    classifier_run_dir: str | Path,
    split: str = "validation",
    tau_grid: list[float] | tuple[float, ...] | None = None,
    localization_checkpoint: str | Path | None = None,
    classifier_checkpoint: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    localization_run_dir = Path(localization_run_dir)
    classifier_run_dir = Path(classifier_run_dir)

    localization_cfg = _load_yaml(localization_run_dir / "config.yaml")
    classifier_cfg = _load_yaml(classifier_run_dir / "config.yaml")

    localization_checkpoint_path = resolve_localization_checkpoint(
        localization_run_dir,
        checkpoint=localization_checkpoint,
    )
    classifier_checkpoint_path = resolve_classifier_checkpoint(
        classifier_run_dir,
        checkpoint=classifier_checkpoint,
    )

    data_root, _checkpoint_root = get_paths(
        device=classifier_cfg.get("paths_device", localization_cfg.get("paths_device", "h200"))
    )
    output_dir = resolve_output_dir(
        localization_run_dir=localization_run_dir,
        classifier_run_dir=classifier_run_dir,
        out_dir=out_dir,
    )
    print(f"[filter-eval] output_dir={output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[filter-eval] device={device}")
    dataset = _build_validation_dataset(
        data_root=data_root,
        localization_cfg=localization_cfg,
        split=split,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    localization_model = _load_localization_model(
        localization_cfg=localization_cfg,
        checkpoint_path=localization_checkpoint_path,
        device=device,
    )
    classifier_model = _load_classifier_model(
        classifier_cfg=classifier_cfg,
        checkpoint_path=classifier_checkpoint_path,
        device=device,
    )

    proposal_rows, gt_by_image = _collect_validation_proposals(
        localization_model=localization_model,
        classifier_model=classifier_model,
        dataset=dataset,
        loader=loader,
        localization_cfg=localization_cfg,
        classifier_cfg=classifier_cfg,
        localization_checkpoint_name=localization_checkpoint_path.name,
        classifier_checkpoint_name=classifier_checkpoint_path.name,
        split=split,
        device=device,
    )
    print(
        f"[filter-eval] collected {len(proposal_rows)} proposals "
        f"across {len(gt_by_image)} images for split={split}"
    )

    tau_values = sorted({float(tau) for tau in (tau_grid or build_default_tau_grid())})
    if not tau_values:
        raise ValueError("tau_grid must contain at least one value")

    matching_radius = float(localization_cfg["matching_radius"])
    matching_mode = str(localization_cfg.get("matching_mode", "optimal"))
    sweep_rows = []
    for tau in tqdm(tau_values, desc="tau sweep", leave=False):
        sweep_rows.append(
            _evaluate_tau(
                proposal_rows=proposal_rows,
                gt_by_image=gt_by_image,
                tau=tau,
                matching_radius=matching_radius,
                matching_mode=matching_mode,
            )
        )
    

    baseline_metrics = _evaluate_tau(
        proposal_rows=proposal_rows,
        gt_by_image=gt_by_image,
        tau=1.0,
        matching_radius=matching_radius,
        matching_mode=matching_mode,
    )
    best_row = max(sweep_rows, key=lambda row: (float(row["f1_macro"]), -float(row["tau"])))
    best_tau = float(best_row["tau"])

    for row in proposal_rows:
        keep = _should_keep(p_mined=float(row["p_mined"]), tau=best_tau)
        row["kept_at_best_tau"] = int(keep)
        row["rejected_at_best_tau"] = int(not keep)

    proposal_csv_path = output_dir / f"proposal_scores_{split}.csv"
    sweep_csv_path = output_dir / f"filter_sweep_{split}.csv"
    summary_json_path = output_dir / f"filter_summary_{split}.json"

    _write_csv(proposal_rows, proposal_csv_path)
    _write_csv(sweep_rows, sweep_csv_path)

    summary = {
        "split": split,
        "device": str(device),
        "localization_run_dir": localization_run_dir.as_posix(),
        "localization_checkpoint": localization_checkpoint_path.name,
        "classifier_run_dir": classifier_run_dir.as_posix(),
        "classifier_checkpoint": classifier_checkpoint_path.name,
        "num_images": len(gt_by_image),
        "num_proposals": len(proposal_rows),
        "baseline": baseline_metrics,
        "best_tau": best_tau,
        "best_filtered": best_row,
        "delta_best_minus_baseline": {
            key: float(best_row[key]) - float(baseline_metrics[key])
            for key in (
                "precision",
                "recall",
                "f1",
                "f1_pos",
                "f1_neg",
                "f1_macro",
            )
        },
        "tau_grid": tau_values,
        "note": (
            "Validation-only development benchmark. Threshold selection and reporting "
            "both use the same split."
        ),
        "artifacts": {
            "proposal_scores_csv": proposal_csv_path.as_posix(),
            "filter_sweep_csv": sweep_csv_path.as_posix(),
            "summary_json": summary_json_path.as_posix(),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
