from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from kiloc.datasets.bcdata import IMAGENET_MEAN, IMAGENET_STD

SUPPORTED_CROP_SIZES = {48, 64, 128}
CLASS_ID_TO_NAME = {
    0: "annotated_positive",
    1: "annotated_negative",
    2: "mined_rexclude30",
}


@dataclass(frozen=True)
class CropRecord:
    sample_id: str
    split: str
    image_path: str
    image_id: str
    x: float
    y: float
    label: int
    source: str
    point_id: str = ""
    pred_id: str = ""
    pred_class: str = ""
    score: float = float("nan")
    mining_tag: str = ""


def _split_from_image_path(image_path: str) -> str:
    parts = Path(image_path).parts
    if len(parts) < 3:
        raise ValueError(f"Expected image_path like images/train/0.png, got {image_path}")
    return parts[1]


def load_split_records(
    *,
    split: str,
    annotated_csv: str | Path,
    mined_csv: str | Path | None = None,
) -> list[CropRecord]:
    records = _load_annotated_records(annotated_csv=annotated_csv, split=split)
    if mined_csv is not None:
        records.extend(_load_mined_records(mined_csv=mined_csv, split=split))
    if not records:
        raise RuntimeError(f"No crop records were loaded for split={split}")
    return records


def _load_annotated_records(*, annotated_csv: str | Path, split: str) -> list[CropRecord]:
    records: list[CropRecord] = []
    annotated_csv = Path(annotated_csv)
    with annotated_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = _split_from_image_path(row["image_path"])
            if row_split != split:
                raise ValueError(
                    f"Annotated table {annotated_csv} mixes splits: expected {split}, found {row_split}"
                )

            source_label = int(row["label"])
            if source_label == 1:
                label = 0
                source = "annotated_positive"
            elif source_label == 0:
                label = 1
                source = "annotated_negative"
            else:
                raise ValueError(f"Unexpected annotated label {source_label} in {annotated_csv}")

            point_id = row.get("point_id", "")
            records.append(
                CropRecord(
                    sample_id=f"ann_{point_id}",
                    split=split,
                    image_path=row["image_path"],
                    image_id=row["image_id"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    label=label,
                    source=source,
                    point_id=point_id,
                )
            )
    return records


def _load_mined_records(*, mined_csv: str | Path, split: str) -> list[CropRecord]:
    records: list[CropRecord] = []
    mined_csv = Path(mined_csv)
    with mined_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = _split_from_image_path(row["image_path"])
            if row_split != split:
                continue

            records.append(
                CropRecord(
                    sample_id=f"mined_{row.get('fold', '')}_{row.get('pred_id', '')}",
                    split=split,
                    image_path=row["image_path"],
                    image_id=row["image_id"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    label=2,
                    source="mined_rexclude30",
                    pred_id=row.get("pred_id", ""),
                    pred_class=row.get("pred_class", ""),
                    score=float(row["score"]) if row.get("score") else float("nan"),
                    mining_tag=row.get("mining_tag", ""),
                )
            )
    return records


def count_by_class(records: list[CropRecord]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {str(class_id): int(counts.get(class_id, 0)) for class_id in sorted(CLASS_ID_TO_NAME)}


def build_image_splits_from_train_records(
    *,
    annotated_train_csv: str | Path,
    mined_train_csv: str | Path | None,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, list[CropRecord]], dict[str, list[str]]]:
    total_ratio = float(train_ratio) + float(validation_ratio) + float(test_ratio)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train/validation/test ratios must sum to 1.0, got {total_ratio:.6f}"
        )

    all_records = load_split_records(
        split="train",
        annotated_csv=annotated_train_csv,
        mined_csv=mined_train_csv,
    )
    image_paths = sorted({record.image_path for record in all_records})
    if len(image_paths) < 3:
        raise ValueError("Need at least 3 train images to create train/validation/test partitions")

    rng = np.random.default_rng(seed)
    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    num_images = len(shuffled)
    num_test = int(round(num_images * test_ratio))
    num_validation = int(round(num_images * validation_ratio))
    num_test = max(1, min(num_test, num_images - 2))
    num_validation = max(1, min(num_validation, num_images - num_test - 1))
    num_train = num_images - num_validation - num_test
    if num_train < 1:
        raise ValueError("Image split produced an empty train set")

    split_to_images = {
        "train": sorted(shuffled[:num_train]),
        "validation": sorted(shuffled[num_train:num_train + num_validation]),
        "test": sorted(shuffled[num_train + num_validation:]),
    }
    image_to_split = {
        image_path: split_name
        for split_name, image_list in split_to_images.items()
        for image_path in image_list
    }

    split_records: dict[str, list[CropRecord]] = {"train": [], "validation": [], "test": []}
    for record in all_records:
        split_records[image_to_split[record.image_path]].append(record)

    return split_records, split_to_images


def build_balanced_sampler(records: list[CropRecord]) -> WeightedRandomSampler:
    class_counts = Counter(record.label for record in records)
    if not class_counts:
        raise ValueError("Cannot build a sampler for an empty record list")

    weights = [1.0 / class_counts[record.label] for record in records]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
    )


def read_image_rgb(
    image_path: str | Path,
    *,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    image_path = Path(image_path)
    cache_key = image_path.as_posix()
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image at {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if cache is not None:
        cache[cache_key] = image_rgb
    return image_rgb


def extract_center_crop_uint8(
    image_rgb: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    crop_size: int,
    resize_hw: tuple[int, int] = (128, 128),
) -> np.ndarray:
    crop_size = int(crop_size)
    half = crop_size // 2

    cx = int(round(center_x))
    cy = int(round(center_y))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - image_rgb.shape[1])
    pad_bottom = max(0, y1 - image_rgb.shape[0])

    if pad_left or pad_top or pad_right or pad_bottom:
        image_rgb = cv2.copyMakeBorder(
            image_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    crop = image_rgb[y0:y1, x0:x1]
    if crop.shape[:2] != (crop_size, crop_size):
        raise RuntimeError(
            f"Crop has shape {crop.shape[:2]}, expected {(crop_size, crop_size)}"
        )

    resize_hw = tuple(int(x) for x in resize_hw)
    if resize_hw != (crop_size, crop_size):
        crop = cv2.resize(
            crop,
            dsize=(resize_hw[1], resize_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return crop


def crop_uint8_to_tensor(
    crop_rgb: np.ndarray,
    *,
    input_normalization: str | None = "imagenet",
) -> torch.Tensor:
    crop_float = crop_rgb.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(crop_float).permute(2, 0, 1).contiguous().float()
    if input_normalization == "imagenet":
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return image_tensor


class ThreeClassCropDataset(Dataset):
    def __init__(
        self,
        *,
        root: str | Path,
        records: list[CropRecord],
        crop_size: int,
        resize_hw: tuple[int, int] = (128, 128),
        input_normalization: str | None = "imagenet",
        cache_images: bool = False,
    ) -> None:
        if crop_size not in SUPPORTED_CROP_SIZES:
            raise ValueError(
                f"crop_size must be one of {sorted(SUPPORTED_CROP_SIZES)}, got {crop_size}"
            )
        self.root = Path(root)
        self.records = records
        self.crop_size = int(crop_size)
        self.resize_hw = tuple(int(x) for x in resize_hw)
        self.input_normalization = input_normalization
        self.cache_images = cache_images
        self._image_cache: dict[str, np.ndarray] = {}

        if not self.records:
            raise ValueError("ThreeClassCropDataset requires at least one record")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, record: CropRecord) -> Path:
        return self.root / record.image_path

    def _read_image_rgb(self, image_path: Path) -> np.ndarray:
        return read_image_rgb(
            image_path,
            cache=self._image_cache if self.cache_images else None,
        )

    def _extract_crop_uint8(self, image_rgb: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
        return extract_center_crop_uint8(
            image_rgb,
            center_x=center_x,
            center_y=center_y,
            crop_size=self.crop_size,
            resize_hw=self.resize_hw,
        )

    def get_crop_uint8(self, idx: int) -> np.ndarray:
        record = self.records[idx]
        image_rgb = self._read_image_rgb(self._resolve_image_path(record))
        return self._extract_crop_uint8(image_rgb=image_rgb, center_x=record.x, center_y=record.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        record = self.records[idx]
        crop_rgb = self.get_crop_uint8(idx)
        image_tensor = crop_uint8_to_tensor(
            crop_rgb,
            input_normalization=self.input_normalization,
        )

        target = torch.tensor(record.label, dtype=torch.long)
        metadata = {
            "dataset_index": idx,
            "sample_id": record.sample_id,
            "split": record.split,
            "image_path": record.image_path,
            "image_id": record.image_id,
            "x": float(record.x),
            "y": float(record.y),
            "true_label": int(record.label),
            "true_label_name": CLASS_ID_TO_NAME[record.label],
            "source": record.source,
            "point_id": record.point_id,
            "pred_id": record.pred_id,
            "source_pred_class": record.pred_class,
            "source_score": float(record.score),
            "mining_tag": record.mining_tag,
            "crop_size": int(self.crop_size),
        }
        return image_tensor, target, metadata
