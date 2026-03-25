from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def read_mined_false_positive_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "fold": int(row["fold"]),
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "pred_id": row["pred_id"],
                    "pred_class": row["pred_class"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "score": float(row["score"]),
                    "score_pos": float(row["score_pos"]),
                    "score_neg": float(row["score_neg"]),
                    "decode_threshold": float(row["decode_threshold"]),
                    "model_checkpoint": row["model_checkpoint"],
                    "raw_rank_in_image": int(row["raw_rank_in_image"]),
                    "nearest_gt_any_class": row["nearest_gt_any_class"],
                    "nearest_gt_any_dist": _parse_optional_float(row["nearest_gt_any_dist"]),
                    "nearest_gt_same_dist": _parse_optional_float(row["nearest_gt_same_dist"]),
                    "nearest_gt_other_dist": _parse_optional_float(row["nearest_gt_other_dist"]),
                    "sameclass_cluster_size": int(row["sameclass_cluster_size"]),
                    "anyclass_cluster_size": int(row["anyclass_cluster_size"]),
                    "cluster_sameclass_id": row["cluster_sameclass_id"],
                    "cluster_anyclass_id": row["cluster_anyclass_id"],
                    "mining_tag": row["mining_tag"],
                    "crop_size_for_review": row["crop_size_for_review"],
                    "review_status": row["review_status"],
                    "review_note": row["review_note"],
                }
            )
    return rows


def group_mined_false_positives_by_image(
    rows: list[dict[str, Any]],
    *,
    min_score: float | None = None,
    pred_classes: tuple[str, ...] | list[str] | None = None,
    max_points_per_image: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if isinstance(pred_classes, str):
        allowed_classes = {pred_classes}
    else:
        allowed_classes = None if pred_classes is None else {str(pred_class) for pred_class in pred_classes}
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        if min_score is not None and float(row["score"]) < float(min_score):
            continue
        if allowed_classes is not None and row["pred_class"] not in allowed_classes:
            continue
        grouped.setdefault(row["image_id"], []).append(row)

    for image_id, image_rows in grouped.items():
        image_rows.sort(
            key=lambda row: (-float(row["score"]), int(row["raw_rank_in_image"]), str(row["pred_id"]))
        )
        if max_points_per_image is not None:
            grouped[image_id] = image_rows[:max_points_per_image]

    return grouped


def summarize_grouped_mined_false_positives(
    grouped_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_class_counts = Counter()
    points_per_image = []

    for image_rows in grouped_rows.values():
        points_per_image.append(len(image_rows))
        for row in image_rows:
            per_class_counts[row["pred_class"]] += 1

    num_points = sum(points_per_image)
    num_images = len(grouped_rows)
    mean_points_per_image = num_points / num_images if num_images > 0 else 0.0

    return {
        "num_images_with_candidates": num_images,
        "num_candidates": num_points,
        "mean_candidates_per_image": mean_points_per_image,
        "max_candidates_per_image": max(points_per_image) if points_per_image else 0,
        "per_class_counts": {str(key): per_class_counts[key] for key in sorted(per_class_counts)},
    }
