from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from kiloc.oof import relation_artifact_dir


FOLD_DIR_RE = re.compile(r"^fold_(\d+)$")
FOLD_RELATION_CSV_RE = re.compile(r"^fold_(\d+)_prediction_relations\.csv$")
TAGGED_FOLD_RELATION_CSV_RE = re.compile(r"^fold_(\d+)_prediction_relations_([A-Za-z0-9_.-]+)\.csv$")


def _image_sort_key(image_id: str) -> tuple[int, str]:
    if image_id.isdigit():
        return (0, f"{int(image_id):08d}")
    return (1, image_id)


def discover_relation_csvs(
    oof_run_dir: str | Path,
    *,
    fold_indices: list[int] | tuple[int, ...] | None = None,
    tag: str | None = None,
) -> list[tuple[int, Path]]:
    oof_run_dir = Path(oof_run_dir)
    discovered: dict[int, Path] = {}

    pattern = "fold_*_prediction_relations.csv" if tag is None else f"fold_*_prediction_relations_{tag}.csv"

    for path in oof_run_dir.glob(pattern):
        match = (
            FOLD_RELATION_CSV_RE.match(path.name)
            if tag is None
            else TAGGED_FOLD_RELATION_CSV_RE.match(path.name)
        )
        if match is None:
            continue
        discovered[int(match.group(1))] = path

    for child in oof_run_dir.iterdir():
        if not child.is_dir():
            continue
        match = FOLD_DIR_RE.match(child.name)
        if match is None:
            continue
        fold_index = int(match.group(1))
        relation_csv_name = (
            f"fold_{fold_index}_prediction_relations.csv"
            if tag is None
            else f"fold_{fold_index}_prediction_relations_{tag}.csv"
        )
        candidate_paths = [
            relation_artifact_dir(child) / relation_csv_name,
            child / relation_csv_name,
        ]
        for relation_csv_path in candidate_paths:
            if relation_csv_path.exists():
                discovered[fold_index] = relation_csv_path
                break

    if not discovered:
        raise FileNotFoundError(f"No fold relation CSVs found under {oof_run_dir}")

    if fold_indices is not None:
        fold_indices = sorted(set(int(fold_index) for fold_index in fold_indices))
        missing = [fold_index for fold_index in fold_indices if fold_index not in discovered]
        if missing:
            raise FileNotFoundError(
                f"Requested folds are missing relation CSVs under {oof_run_dir}: {missing}"
            )
        return [(fold_index, discovered[fold_index]) for fold_index in fold_indices]

    return sorted(discovered.items())


def build_mined_false_positive_rows(
    relation_rows: list[dict[str, Any]],
    *,
    mining_tag: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in relation_rows:
        if row["filter_reason"] != "kept_for_mining":
            continue

        rows.append(
            {
                "fold": row["fold"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "pred_id": row["pred_id"],
                "pred_class": row["pred_class"],
                "x": row["x"],
                "y": row["y"],
                "score": row["score"],
                "score_pos": row["score_pos"],
                "score_neg": row["score_neg"],
                "decode_threshold": row["decode_threshold"],
                "model_checkpoint": row["model_checkpoint"],
                "raw_rank_in_image": row["raw_rank_in_image"],
                "nearest_gt_any_class": row["nearest_gt_any_class"],
                "nearest_gt_any_dist": row["nearest_gt_any_dist"],
                "nearest_gt_same_dist": row["nearest_gt_same_dist"],
                "nearest_gt_other_dist": row["nearest_gt_other_dist"],
                "sameclass_cluster_size": row["n_preds_sameclass_within_r_cluster"],
                "anyclass_cluster_size": row["n_preds_anyclass_within_r_interclass"],
                "cluster_sameclass_id": row["cluster_sameclass_id"],
                "cluster_anyclass_id": row["cluster_anyclass_id"],
                "mining_tag": mining_tag,
                "crop_size_for_review": "",
                "review_status": "",
                "review_note": "",
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["fold"]),
            _image_sort_key(str(row["image_id"])),
            str(row["pred_class"]),
            int(row["raw_rank_in_image"]),
            str(row["pred_id"]),
        )
    )
    return rows


def summarize_mined_false_positive_rows(
    rows: list[dict[str, Any]],
    *,
    relation_csvs: list[tuple[int, Path]] | None = None,
) -> dict[str, Any]:
    per_fold_counts = Counter()
    per_class_counts = Counter()
    per_fold_class_counts = Counter()
    image_ids = set()

    for row in rows:
        fold_index = int(row["fold"])
        pred_class = str(row["pred_class"])
        per_fold_counts[fold_index] += 1
        per_class_counts[pred_class] += 1
        per_fold_class_counts[(fold_index, pred_class)] += 1
        image_ids.add((fold_index, str(row["image_id"])))

    summary: dict[str, Any] = {
        "num_mined_rows": len(rows),
        "num_fold_image_pairs": len(image_ids),
        "per_fold_counts": {str(k): per_fold_counts[k] for k in sorted(per_fold_counts)},
        "per_class_counts": {str(k): per_class_counts[k] for k in sorted(per_class_counts)},
        "per_fold_class_counts": {
            f"fold_{fold_index}_{pred_class}": per_fold_class_counts[(fold_index, pred_class)]
            for fold_index, pred_class in sorted(per_fold_class_counts)
        },
    }

    if relation_csvs is not None:
        summary["relation_csvs"] = {
            str(fold_index): str(path) for fold_index, path in relation_csvs
        }

    return summary


def write_mined_false_positive_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
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
        "nearest_gt_any_class",
        "nearest_gt_any_dist",
        "nearest_gt_same_dist",
        "nearest_gt_other_dist",
        "sameclass_cluster_size",
        "anyclass_cluster_size",
        "cluster_sameclass_id",
        "cluster_anyclass_id",
        "mining_tag",
        "crop_size_for_review",
        "review_status",
        "review_note",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_mined_false_positive_summary(
    summary: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
