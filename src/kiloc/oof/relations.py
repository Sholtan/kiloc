from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    points_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    points_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)

    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a), len(points_b)), dtype=np.float32)

    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2, dtype=np.float32)).astype(np.float32, copy=False)


def _connected_components(points_xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    n = len(points_xy)
    if n == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    if radius <= 0:
        labels = np.arange(n, dtype=np.int64)
        sizes = np.ones(n, dtype=np.int64)
        return labels, sizes

    dist = _distance_matrix(points_xy, points_xy)
    adjacency = dist <= float(radius)

    labels = np.full(n, -1, dtype=np.int64)
    component_sizes: list[int] = []
    next_label = 0

    for start in range(n):
        if labels[start] != -1:
            continue

        stack = [start]
        labels[start] = next_label
        members = [start]

        while stack:
            current = stack.pop()
            neighbors = np.where(adjacency[current])[0]
            for neighbor in neighbors:
                if labels[neighbor] != -1:
                    continue
                labels[neighbor] = next_label
                stack.append(int(neighbor))
                members.append(int(neighbor))

        component_sizes.append(len(members))
        next_label += 1

    sizes = np.array([component_sizes[label] for label in labels], dtype=np.int64)
    return labels, sizes


def _neighbor_counts(points_xy: np.ndarray, radius: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(points_xy) == 0:
        return np.empty((0,), dtype=np.int64)
    if radius <= 0:
        return np.ones(len(points_xy), dtype=np.int64)

    dist = _distance_matrix(points_xy, points_xy)
    return (dist <= float(radius)).sum(axis=1).astype(np.int64, copy=False)


def _format_gt_id(gt_class: str, gt_index: int) -> str:
    return f"{gt_class}_{gt_index:04d}"


def _load_points_from_h5(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        coords = f["coordinates"][:]

    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return _empty_points()
    return coords.reshape(-1, 2)


def load_gt_by_image_from_image_paths(
    data_root: str | Path,
    image_paths: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    data_root = Path(data_root)
    gt_by_image: dict[str, dict[str, Any]] = {}

    for rel_image_path in image_paths:
        rel_path = Path(rel_image_path)
        if len(rel_path.parts) < 3:
            raise ValueError(
                f"Expected image path like images/<split>/<name>.png, got {rel_image_path!r}"
            )

        split = rel_path.parts[1]
        image_id = rel_path.stem
        pos_ann_path = data_root / "annotations" / split / "positive" / f"{image_id}.h5"
        neg_ann_path = data_root / "annotations" / split / "negative" / f"{image_id}.h5"

        gt_by_image[image_id] = {
            "image_id": image_id,
            "image_path": rel_path.as_posix(),
            "split": split,
            "pos_points": _load_points_from_h5(pos_ann_path),
            "neg_points": _load_points_from_h5(neg_ann_path),
        }

    return gt_by_image


def read_raw_prediction_csv(path: str | Path) -> list[dict[str, Any]]:
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
                }
            )
    return rows


def _parse_optional_float(value: str) -> float | str:
    if value == "":
        return ""
    return float(value)


def read_relation_csv(path: str | Path) -> list[dict[str, Any]]:
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
                    "matched_gt_id": row["matched_gt_id"],
                    "matched_gt_class": row["matched_gt_class"],
                    "matched_dist": _parse_optional_float(row["matched_dist"]),
                    "nearest_gt_any_id": row["nearest_gt_any_id"],
                    "nearest_gt_any_class": row["nearest_gt_any_class"],
                    "nearest_gt_any_dist": _parse_optional_float(row["nearest_gt_any_dist"]),
                    "nearest_gt_same_id": row["nearest_gt_same_id"],
                    "nearest_gt_same_dist": _parse_optional_float(row["nearest_gt_same_dist"]),
                    "nearest_gt_other_id": row["nearest_gt_other_id"],
                    "nearest_gt_other_dist": _parse_optional_float(row["nearest_gt_other_dist"]),
                    "n_preds_sameclass_within_r_cluster": int(row["n_preds_sameclass_within_r_cluster"]),
                    "n_preds_anyclass_within_r_interclass": int(row["n_preds_anyclass_within_r_interclass"]),
                    "cluster_sameclass_id": row["cluster_sameclass_id"],
                    "cluster_anyclass_id": row["cluster_anyclass_id"],
                    "filter_status": row["filter_status"],
                    "filter_reason": row["filter_reason"],
                }
            )
    return rows


def _nearest_info(
    pred_point: np.ndarray,
    gt_points: np.ndarray,
    gt_class: str | None,
) -> tuple[str, str, str | float]:
    gt_points = np.asarray(gt_points, dtype=np.float32).reshape(-1, 2)
    if len(gt_points) == 0:
        return "", "", ""

    dists = _distance_matrix(pred_point.reshape(1, 2), gt_points)[0]
    best_idx = int(dists.argmin())
    gt_id = _format_gt_id(gt_class if gt_class is not None else "gt", best_idx)
    gt_class_value = gt_class if gt_class is not None else ""
    return gt_id, gt_class_value, float(dists[best_idx])


def _nearest_any_info(
    pred_point: np.ndarray,
    pos_points: np.ndarray,
    neg_points: np.ndarray,
) -> tuple[str, str, str | float]:
    best_id = ""
    best_class = ""
    best_dist: float | None = None

    for gt_class, gt_points in (("pos", pos_points), ("neg", neg_points)):
        gt_id, _, gt_dist = _nearest_info(pred_point, gt_points, gt_class)
        if gt_id == "":
            continue
        assert isinstance(gt_dist, float)
        if best_dist is None or gt_dist < best_dist:
            best_id = gt_id
            best_class = gt_class
            best_dist = gt_dist

    return best_id, best_class, "" if best_dist is None else float(best_dist)


def _greedy_class_aware_matches(
    rows: list[dict[str, Any]],
    gt_points: np.ndarray,
    gt_class: str,
    matching_radius: float,
) -> dict[str, tuple[str, float]]:
    if len(rows) == 0 or len(gt_points) == 0:
        return {}

    pred_points = np.array([[row["x"], row["y"]] for row in rows], dtype=np.float32)
    dist = _distance_matrix(pred_points, gt_points)
    ordered_rows = sorted(rows, key=lambda row: (-row["score"], row["raw_rank_in_image"]))

    matched_gt_indices: set[int] = set()
    matches: dict[str, tuple[str, float]] = {}
    row_index_by_pred_id = {row["pred_id"]: idx for idx, row in enumerate(rows)}

    for row in ordered_rows:
        pred_idx = row_index_by_pred_id[row["pred_id"]]
        if dist.shape[1] == 0:
            continue

        best_gt_idx = int(dist[pred_idx].argmin())
        best_dist = float(dist[pred_idx, best_gt_idx])
        if best_dist <= matching_radius and best_gt_idx not in matched_gt_indices:
            matched_gt_indices.add(best_gt_idx)
            matches[row["pred_id"]] = (_format_gt_id(gt_class, best_gt_idx), best_dist)

    return matches


def _build_image_relation_rows(
    *,
    image_rows: list[dict[str, Any]],
    image_gt: dict[str, Any],
    tau_mine: tuple[float, float],
    matching_radius: float,
    r_exclude_any: float,
    r_cluster_same: float,
    r_interclass_conflict: float,
) -> list[dict[str, Any]]:
    if not image_rows:
        return []

    pos_gt = np.asarray(image_gt["pos_points"], dtype=np.float32).reshape(-1, 2)
    neg_gt = np.asarray(image_gt["neg_points"], dtype=np.float32).reshape(-1, 2)

    pos_rows = [row for row in image_rows if row["pred_class"] == "pos"]
    neg_rows = [row for row in image_rows if row["pred_class"] == "neg"]

    pos_matches = _greedy_class_aware_matches(pos_rows, pos_gt, "pos", matching_radius)
    neg_matches = _greedy_class_aware_matches(neg_rows, neg_gt, "neg", matching_radius)
    matches = {**pos_matches, **neg_matches}

    pred_points_all = np.array([[row["x"], row["y"]] for row in image_rows], dtype=np.float32)
    pred_classes_all = np.array([row["pred_class"] for row in image_rows], dtype=object)

    any_counts = _neighbor_counts(pred_points_all, r_interclass_conflict)
    any_labels, _ = _connected_components(pred_points_all, r_interclass_conflict)
    index_by_pred_id = {row["pred_id"]: idx for idx, row in enumerate(image_rows)}

    same_counts_by_pred_id: dict[str, int] = {}
    same_cluster_by_pred_id: dict[str, str] = {}

    for pred_class in ("pos", "neg"):
        class_rows = [row for row in image_rows if row["pred_class"] == pred_class]
        class_points = np.array([[row["x"], row["y"]] for row in class_rows], dtype=np.float32)
        class_counts = _neighbor_counts(class_points, r_cluster_same)
        class_labels, _ = _connected_components(class_points, r_cluster_same)

        for local_idx, row in enumerate(class_rows):
            same_counts_by_pred_id[row["pred_id"]] = int(class_counts[local_idx])
            same_cluster_by_pred_id[row["pred_id"]] = (
                f"{row['image_id']}_{pred_class}_cluster_{int(class_labels[local_idx]):04d}"
            )

    opposite_within_radius: dict[str, int] = {row["pred_id"]: 0 for row in image_rows}
    if len(pos_rows) > 0 and len(neg_rows) > 0 and r_interclass_conflict > 0:
        pos_points = np.array([[row["x"], row["y"]] for row in pos_rows], dtype=np.float32)
        neg_points = np.array([[row["x"], row["y"]] for row in neg_rows], dtype=np.float32)
        pos_neg_dist = _distance_matrix(pos_points, neg_points)
        pos_conflicts = (pos_neg_dist <= r_interclass_conflict).sum(axis=1).astype(np.int64)
        neg_conflicts = (pos_neg_dist <= r_interclass_conflict).sum(axis=0).astype(np.int64)

        for idx, row in enumerate(pos_rows):
            opposite_within_radius[row["pred_id"]] = int(pos_conflicts[idx])
        for idx, row in enumerate(neg_rows):
            opposite_within_radius[row["pred_id"]] = int(neg_conflicts[idx])

    provisional: list[dict[str, Any]] = []
    duplicate_candidates_by_cluster: dict[str, list[dict[str, Any]]] = {}

    for row in image_rows:
        pred_point = np.array([row["x"], row["y"]], dtype=np.float32)
        pred_class = row["pred_class"]
        same_gt = pos_gt if pred_class == "pos" else neg_gt
        other_gt = neg_gt if pred_class == "pos" else pos_gt
        tau = tau_mine[0] if pred_class == "pos" else tau_mine[1]

        nearest_any_id, nearest_any_class, nearest_any_dist = _nearest_any_info(
            pred_point,
            pos_points=pos_gt,
            neg_points=neg_gt,
        )
        nearest_same_id, _, nearest_same_dist = _nearest_info(pred_point, same_gt, pred_class)
        other_gt_class = "neg" if pred_class == "pos" else "pos"
        nearest_other_id, _, nearest_other_dist = _nearest_info(pred_point, other_gt, other_gt_class)

        matched_gt_id = ""
        matched_gt_class = ""
        matched_dist: str | float = ""
        if row["pred_id"] in matches:
            matched_gt_id, matched_dist_value = matches[row["pred_id"]]
            matched_gt_id = matched_gt_id
            matched_gt_class = pred_class
            matched_dist = matched_dist_value

        base = {
            **row,
            "matched_gt_id": matched_gt_id,
            "matched_gt_class": matched_gt_class,
            "matched_dist": matched_dist,
            "nearest_gt_any_id": nearest_any_id,
            "nearest_gt_any_class": nearest_any_class,
            "nearest_gt_any_dist": nearest_any_dist,
            "nearest_gt_same_id": nearest_same_id,
            "nearest_gt_same_dist": nearest_same_dist,
            "nearest_gt_other_id": nearest_other_id,
            "nearest_gt_other_dist": nearest_other_dist,
            "n_preds_sameclass_within_r_cluster": same_counts_by_pred_id[row["pred_id"]],
            "n_preds_anyclass_within_r_interclass": int(any_counts[index_by_pred_id[row["pred_id"]]]),
            "cluster_sameclass_id": same_cluster_by_pred_id[row["pred_id"]],
            "cluster_anyclass_id": f"{row['image_id']}_any_cluster_{int(any_labels[index_by_pred_id[row['pred_id']]]):04d}",
        }

        if matched_gt_id != "":
            base["filter_status"] = "matched"
            base["filter_reason"] = "matched"
        elif row["score"] < tau:
            base["filter_status"] = "rejected"
            base["filter_reason"] = "below_score_threshold"
        elif nearest_any_dist != "" and float(nearest_any_dist) <= r_exclude_any:
            base["filter_status"] = "rejected"
            base["filter_reason"] = "near_gt_any"
        elif opposite_within_radius[row["pred_id"]] > 0:
            base["filter_status"] = "rejected"
            base["filter_reason"] = "interclass_conflict"
        else:
            base["filter_status"] = "candidate"
            base["filter_reason"] = "candidate"
            duplicate_candidates_by_cluster.setdefault(base["cluster_sameclass_id"], []).append(base)

        provisional.append(base)

    for cluster_rows in duplicate_candidates_by_cluster.values():
        cluster_rows.sort(key=lambda item: (-item["score"], item["raw_rank_in_image"], item["pred_id"]))
        keep_pred_id = cluster_rows[0]["pred_id"]
        for row in cluster_rows:
            if row["pred_id"] == keep_pred_id:
                row["filter_status"] = "kept"
                row["filter_reason"] = "kept_for_mining"
            else:
                row["filter_status"] = "rejected"
                row["filter_reason"] = "sameclass_duplicate"

    provisional.sort(key=lambda row: (row["image_id"], row["pred_class"], row["raw_rank_in_image"]))
    return provisional


def build_relation_rows(
    *,
    raw_rows: list[dict[str, Any]],
    gt_by_image: dict[str, dict[str, Any]],
    tau_mine: tuple[float, float],
    matching_radius: float,
    r_exclude_any: float,
    r_cluster_same: float,
    r_interclass_conflict: float,
) -> list[dict[str, Any]]:
    rows_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        rows_by_image.setdefault(row["image_id"], []).append(row)

    relation_rows: list[dict[str, Any]] = []
    for image_id, image_rows in rows_by_image.items():
        if image_id not in gt_by_image:
            raise KeyError(f"Missing GT annotations for image_id={image_id}")
        relation_rows.extend(
            _build_image_relation_rows(
                image_rows=image_rows,
                image_gt=gt_by_image[image_id],
                tau_mine=tau_mine,
                matching_radius=matching_radius,
                r_exclude_any=r_exclude_any,
                r_cluster_same=r_cluster_same,
                r_interclass_conflict=r_interclass_conflict,
            )
        )

    return relation_rows


def summarize_relation_rows(
    relation_rows: list[dict[str, Any]],
    *,
    total_holdout_images: int | None = None,
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    image_paths = {row["image_path"] for row in relation_rows}

    for row in relation_rows:
        reason_counts[row["filter_reason"]] = reason_counts.get(row["filter_reason"], 0) + 1
        status_counts[row["filter_status"]] = status_counts.get(row["filter_status"], 0) + 1

    summary = {
        "num_relation_rows": len(relation_rows),
        "num_images_with_predictions": len(image_paths),
        "filter_reason_counts": reason_counts,
        "filter_status_counts": status_counts,
    }
    if total_holdout_images is not None:
        summary["total_holdout_images"] = total_holdout_images
        summary["num_images_without_predictions"] = max(total_holdout_images - len(image_paths), 0)

    return summary


def write_relation_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    fieldnames = [
        "fold",
        "image_id",
        "image_path",
        "pred_id",
        "pred_class",
        "x",
        "y",
        "score",
        "score_pos",
        "score_neg",
        "decode_threshold",
        "model_checkpoint",
        "raw_rank_in_image",
        "matched_gt_id",
        "matched_gt_class",
        "matched_dist",
        "nearest_gt_any_id",
        "nearest_gt_any_class",
        "nearest_gt_any_dist",
        "nearest_gt_same_id",
        "nearest_gt_same_dist",
        "nearest_gt_other_id",
        "nearest_gt_other_dist",
        "n_preds_sameclass_within_r_cluster",
        "n_preds_anyclass_within_r_interclass",
        "cluster_sameclass_id",
        "cluster_anyclass_id",
        "filter_status",
        "filter_reason",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_relation_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
