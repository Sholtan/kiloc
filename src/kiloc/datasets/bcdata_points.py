from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .bcdata import IMAGENET_MEAN, IMAGENET_STD


class BCDataPointDataset(Dataset):
    """
    Dataset of point-centered classification crops for BCData.

    Each row in the CSV corresponds to one annotated cell point and must contain:
        point_id,image_path,image_id,x,y,label

    Notes
    -----
    - `image_path` is expected to be relative to `data_root`.
    - Border regions outside the image are filled with `background_rgb`.
    - `crop_size` is the side length in original image pixels.
    - `resize_to` is optional classifier input size. If None, no resize is applied.
    - `label` is returned as torch.long (0/1).
    """

    def __init__(
        self,
        data_root: str | Path,
        points_csv: str | Path,
        crop_size: int,
        resize_to: int | None = None,
        image_transform: Callable | None = None,
        input_normalization: str | None = "imagenet",
        background_rgb: tuple[int, int, int] = (240, 240, 240),
    ) -> None:
        self.data_root = Path(data_root)
        self.points_csv = Path(points_csv)
        self.crop_size = int(crop_size)
        self.resize_to = int(resize_to) if resize_to is not None else None
        self.image_transform = image_transform
        self.input_normalization = input_normalization
        self.background_rgb = tuple(int(v) for v in background_rgb)

        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {self.crop_size}")

        if self.resize_to is not None and self.resize_to <= 0:
            raise ValueError(f"resize_to must be positive, got {self.resize_to}")

        if len(self.background_rgb) != 3:
            raise ValueError(
                f"background_rgb must have 3 values, got {self.background_rgb}"
            )

        if not self.points_csv.exists():
            raise FileNotFoundError(f"Points CSV not found: {self.points_csv}")

        self.rows = self._load_rows(self.points_csv)

        # Small cache helps when CSV rows are grouped by image.
        self._cached_image_path: Path | None = None
        self._cached_image_rgb: np.ndarray | None = None

    def _load_rows(self, csv_path: Path) -> list[dict[str, Any]]:
        required_columns = {"point_id", "image_path", "image_id", "x", "y", "label"}

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")

            missing = required_columns.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV {csv_path} is missing required columns: {sorted(missing)}"
                )

            rows: list[dict[str, Any]] = []
            for row in reader:
                image_rel = row["image_path"]
                image_path = self.data_root / image_rel

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image referenced by CSV does not exist: {image_path}"
                    )

                image_id_raw = row["image_id"]
                image_id: int | str
                image_id = int(image_id_raw) if image_id_raw.isdigit() else image_id_raw

                rows.append(
                    {
                        "point_id": int(row["point_id"]),
                        "image_path": image_path,
                        "image_rel_path": image_rel,
                        "image_id": image_id,
                        "x": int(row["x"]),
                        "y": int(row["y"]),
                        "label": int(row["label"]),
                    }
                )

        if not rows:
            raise RuntimeError(f"No rows found in CSV: {csv_path}")

        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _read_image_rgb(self, image_path: Path) -> np.ndarray:
        if self._cached_image_path == image_path and self._cached_image_rgb is not None:
            return self._cached_image_rgb

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self._cached_image_path = image_path
        self._cached_image_rgb = image_rgb
        return image_rgb

    def _crop_with_padding(
        self,
        image_rgb: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> tuple[np.ndarray, int, int]:
        """
        Returns
        -------
        crop_rgb : np.ndarray
            Shape (crop_size, crop_size, 3), dtype uint8.
        local_x, local_y : int
            Point coordinates inside the crop before optional resize.
        """
        h, w, c = image_rgb.shape
        if c != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got {image_rgb.shape}")

        half = self.crop_size // 2
        x0 = center_x - half
        y0 = center_y - half
        x1 = x0 + self.crop_size
        y1 = y0 + self.crop_size

        crop = np.empty((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        crop[...] = np.asarray(self.background_rgb, dtype=np.uint8)

        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x1)
        src_y1 = min(h, y1)

        if src_x1 > src_x0 and src_y1 > src_y0:
            dst_x0 = src_x0 - x0
            dst_y0 = src_y0 - y0
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            crop[dst_y0:dst_y1, dst_x0:dst_x1] = image_rgb[src_y0:src_y1, src_x0:src_x1]

        local_x = center_x - x0
        local_y = center_y - y0
        return crop, local_x, local_y

    def _maybe_resize(
        self,
        crop_rgb: np.ndarray,
        local_x: float,
        local_y: float,
    ) -> tuple[np.ndarray, float, float]:
        if self.resize_to is None or self.resize_to == self.crop_size:
            return crop_rgb, float(local_x), float(local_y)

        interp = cv2.INTER_AREA if self.resize_to < self.crop_size else cv2.INTER_LINEAR
        resized = cv2.resize(
            crop_rgb,
            (self.resize_to, self.resize_to),
            interpolation=interp,
        )

        scale = self.resize_to / float(self.crop_size)
        return resized, local_x * scale, local_y * scale

    def _apply_image_transform(self, image_rgb: np.ndarray) -> np.ndarray | torch.Tensor:
        if self.image_transform is None:
            return image_rgb

        try:
            transformed = self.image_transform(image=image_rgb)
        except TypeError:
            transformed = self.image_transform(image_rgb)

        if isinstance(transformed, dict):
            if "image" not in transformed:
                raise KeyError("Transform returned dict without 'image' key")
            return transformed["image"]

        return transformed

    def _to_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image_t = image.float()
            if image_t.ndim != 3:
                raise ValueError(f"Expected 3D image tensor, got shape {tuple(image_t.shape)}")

            # Convert HWC -> CHW if needed.
            if image_t.shape[0] != 3 and image_t.shape[-1] == 3:
                image_t = image_t.permute(2, 0, 1).contiguous()

            if image_t.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got shape {tuple(image_t.shape)}")

            if image_t.max() > 1.0:
                image_t = image_t / 255.0

            return image_t

        image_np = np.asarray(image)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image_np.shape}")

        image_np = image_np.astype(np.float32)
        if image_np.max() > 1.0:
            image_np /= 255.0

        return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[idx]

        image_rgb = self._read_image_rgb(row["image_path"])
        crop_rgb, local_x, local_y = self._crop_with_padding(
            image_rgb=image_rgb,
            center_x=row["x"],
            center_y=row["y"],
        )
        crop_rgb, local_x, local_y = self._maybe_resize(crop_rgb, local_x, local_y)
        crop = self._apply_image_transform(crop_rgb)
        image_tensor = self._to_tensor(crop)

        if self.input_normalization == "imagenet":
            image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        elif self.input_normalization is not None:
            raise ValueError(
                f"Unsupported input_normalization={self.input_normalization!r}"
            )

        label_tensor = torch.tensor(row["label"], dtype=torch.long)

        meta = {
            "point_id": row["point_id"],
            "image_id": row["image_id"],
            "image_path": row["image_rel_path"],
            "x": row["x"],
            "y": row["y"],
            "local_x": local_x,
            "local_y": local_y,
            "crop_size": self.crop_size,
            "resize_to": self.resize_to,
        }

        return image_tensor, label_tensor, meta

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        images, labels, metas = zip(*batch)
        return torch.stack(images, dim=0), torch.stack(labels, dim=0), list(metas)