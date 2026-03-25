from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: str
    num_pos: int
    num_neg: int
    num_cells: int


@dataclass(frozen=True)
class FoldAssignment:
    fold: int
    image_id: str
    image_path: str
    num_pos: int
    num_neg: int
    num_cells: int


def _natural_image_key(image_id: str) -> tuple[int, int | str]:
    return (0, int(image_id)) if image_id.isdigit() else (1, image_id)


def _load_num_points(ann_path: Path) -> int:
    import h5py

    with h5py.File(ann_path, "r") as f:
        coords = f["coordinates"][:]
    return int(len(coords))


def scan_bcdata_images(
    data_root: str | Path,
    split: str = "train",
) -> list[ImageRecord]:
    data_root = Path(data_root)
    image_dir = data_root / "images" / split
    pos_ann_dir = data_root / "annotations" / split / "positive"
    neg_ann_dir = data_root / "annotations" / split / "negative"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not pos_ann_dir.exists():
        raise FileNotFoundError(f"Missing positive annotation directory: {pos_ann_dir}")
    if not neg_ann_dir.exists():
        raise FileNotFoundError(f"Missing negative annotation directory: {neg_ann_dir}")

    image_paths = sorted(image_dir.glob("*.png"), key=lambda path: _natural_image_key(path.stem))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    records: list[ImageRecord] = []
    for image_path in image_paths:
        image_id = image_path.stem
        pos_ann_path = pos_ann_dir / f"{image_id}.h5"
        neg_ann_path = neg_ann_dir / f"{image_id}.h5"

        if not pos_ann_path.exists():
            raise FileNotFoundError(f"Missing positive annotation file: {pos_ann_path}")
        if not neg_ann_path.exists():
            raise FileNotFoundError(f"Missing negative annotation file: {neg_ann_path}")

        num_pos = _load_num_points(pos_ann_path)
        num_neg = _load_num_points(neg_ann_path)

        records.append(
            ImageRecord(
                image_id=image_id,
                image_path=image_path.relative_to(data_root).as_posix(),
                num_pos=num_pos,
                num_neg=num_neg,
                num_cells=num_pos + num_neg,
            )
        )

    return records


def _fold_imbalance_score(fold_stats: list[dict[str, int]]) -> tuple[int, int, int, int]:
    return (
        max(stats["num_cells"] for stats in fold_stats) - min(stats["num_cells"] for stats in fold_stats),
        max(stats["num_pos"] for stats in fold_stats) - min(stats["num_pos"] for stats in fold_stats),
        max(stats["num_neg"] for stats in fold_stats) - min(stats["num_neg"] for stats in fold_stats),
        max(stats["num_images"] for stats in fold_stats) - min(stats["num_images"] for stats in fold_stats),
    )


def build_balanced_image_folds(
    records: list[ImageRecord],
    num_folds: int = 5,
    seed: int = 41,
) -> list[FoldAssignment]:
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")
    if len(records) < num_folds:
        raise ValueError(
            f"Need at least as many images as folds, got {len(records)} images and {num_folds} folds"
        )

    shuffled_records = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled_records)
    shuffled_records.sort(
        key=lambda record: (record.num_cells, record.num_pos, record.num_neg, record.image_id),
        reverse=True,
    )

    fold_stats = [
        {"num_images": 0, "num_pos": 0, "num_neg": 0, "num_cells": 0}
        for _ in range(num_folds)
    ]

    assignments: list[FoldAssignment] = []

    for record in shuffled_records:
        best_fold = 0
        best_score: tuple[int, int, int, int] | None = None

        for fold_idx in range(num_folds):
            candidate_stats = [stats.copy() for stats in fold_stats]
            candidate_stats[fold_idx]["num_images"] += 1
            candidate_stats[fold_idx]["num_pos"] += record.num_pos
            candidate_stats[fold_idx]["num_neg"] += record.num_neg
            candidate_stats[fold_idx]["num_cells"] += record.num_cells

            score = _fold_imbalance_score(candidate_stats)
            if best_score is None or score < best_score:
                best_fold = fold_idx
                best_score = score

        fold_stats[best_fold]["num_images"] += 1
        fold_stats[best_fold]["num_pos"] += record.num_pos
        fold_stats[best_fold]["num_neg"] += record.num_neg
        fold_stats[best_fold]["num_cells"] += record.num_cells

        assignments.append(
            FoldAssignment(
                fold=best_fold,
                image_id=record.image_id,
                image_path=record.image_path,
                num_pos=record.num_pos,
                num_neg=record.num_neg,
                num_cells=record.num_cells,
            )
        )

    assignments.sort(key=lambda assignment: (assignment.fold, _natural_image_key(assignment.image_id)))
    return assignments


def summarize_fold_assignments(
    assignments: list[FoldAssignment],
    num_folds: int,
) -> list[dict[str, int]]:
    summary = [
        {"fold": fold_idx, "num_images": 0, "num_pos": 0, "num_neg": 0, "num_cells": 0}
        for fold_idx in range(num_folds)
    ]

    for assignment in assignments:
        fold_stats = summary[assignment.fold]
        fold_stats["num_images"] += 1
        fold_stats["num_pos"] += assignment.num_pos
        fold_stats["num_neg"] += assignment.num_neg
        fold_stats["num_cells"] += assignment.num_cells

    return summary


def _write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_fold_manifests(
    assignments: list[FoldAssignment],
    out_dir: str | Path,
    num_folds: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments_sorted = sorted(
        assignments,
        key=lambda assignment: (assignment.fold, _natural_image_key(assignment.image_id)),
    )

    with (out_dir / "fold_assignments.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "image_id", "image_path", "num_pos", "num_neg", "num_cells"],
        )
        writer.writeheader()
        writer.writerows(asdict(assignment) for assignment in assignments_sorted)

    summary = summarize_fold_assignments(assignments_sorted, num_folds=num_folds)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    all_image_paths = sorted(
        [assignment.image_path for assignment in assignments_sorted],
        key=lambda image_path: _natural_image_key(Path(image_path).stem),
    )
    _write_lines(out_dir / "all_images.txt", all_image_paths)

    for fold_idx in range(num_folds):
        holdout_paths = sorted(
            [assignment.image_path for assignment in assignments_sorted if assignment.fold == fold_idx],
            key=lambda image_path: _natural_image_key(Path(image_path).stem),
        )
        train_paths = sorted(
            [assignment.image_path for assignment in assignments_sorted if assignment.fold != fold_idx],
            key=lambda image_path: _natural_image_key(Path(image_path).stem),
        )

        _write_lines(out_dir / f"fold_{fold_idx}_holdout_images.txt", holdout_paths)
        _write_lines(out_dir / f"fold_{fold_idx}_train_images.txt", train_paths)


def load_image_ids(path: str | Path) -> set[str]:
    image_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            if "/" in value or "\\" in value or value.endswith(".png"):
                image_ids.add(Path(value).stem)
            else:
                image_ids.add(value)
    return image_ids
