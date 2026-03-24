#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kiloc.datasets import BCDataDataset
from kiloc.utils.config import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw crop boxes around all BCData annotations on original images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a YAML mapping, got: {type(cfg)}")
    return cfg


def require(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


def center_to_box(x: int, y: int, crop_size: int) -> tuple[int, int, int, int]:
    half = crop_size // 2
    x1 = x - half
    y1 = y - half
    x2 = x1 + crop_size - 1
    y2 = y1 + crop_size - 1
    return x1, y1, x2, y2


def clip_box(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def draw_boxes(
    image_bgr: np.ndarray,
    points_xy: np.ndarray,
    crop_size: int,
    thickness: int,
    draw_centers: bool,
    center_radius: int,
) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    red = (0, 0, 255)        # BGR
    yellow = (0, 255, 255)   # BGR

    for pt in points_xy:
        x = int(pt[0])
        y = int(pt[1])

        x1, y1, x2, y2 = center_to_box(x, y, crop_size)
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, width=w, height=h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), red, thickness=thickness)

        if draw_centers:
            cv2.circle(vis, (x, y), center_radius, yellow, -1)

    return vis


def select_indices(
    n_total: int,
    n_select: int,
    random_sample: bool,
    seed: int,
) -> list[int]:
    n_select = min(n_select, n_total)
    indices = list(range(n_total))
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[:n_select]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = str(require(cfg, "device"))
    split = str(cfg.get("split", "train"))
    crop_size = int(require(cfg, "crop_size"))
    num_images = int(cfg.get("num_images", 100))
    random_sample = bool(cfg.get("random_sample", True))
    seed = int(cfg.get("seed", 42))
    thickness = int(cfg.get("thickness", 1))
    draw_centers = bool(cfg.get("draw_centers", False))
    center_radius = int(cfg.get("center_radius", 1))
    write_manifest = bool(cfg.get("write_manifest", True))

    data_subdir = str(cfg.get("data_subdir", "")).strip()
    out_dir_cfg = str(require(cfg, "out_dir"))

    data_root, checkpoint_dir = get_paths(device)

    dataset_root = data_root / data_subdir if data_subdir else data_root

    out_dir = Path(out_dir_cfg)
    if not out_dir.is_absolute():
        out_dir = checkpoint_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BCDataDataset(
        root=dataset_root,
        split=split,
        target_transform=lambda pts: None,
        joint_transform=None,
        input_normalization=None,
    )

    selected_indices = select_indices(
        n_total=len(dataset),
        n_select=num_images,
        random_sample=random_sample,
        seed=seed,
    )

    manifest_rows: list[dict[str, Any]] = []

    for save_idx, ds_idx in enumerate(selected_indices):
        img_path, pos_ann_path, neg_ann_path = dataset.samples[ds_idx]

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        pos_pts = dataset._load_points(pos_ann_path)
        neg_pts = dataset._load_points(neg_ann_path)
        all_pts = np.concatenate([pos_pts, neg_pts], axis=0)

        vis = draw_boxes(
            image_bgr=image_bgr,
            points_xy=all_pts,
            crop_size=crop_size,
            thickness=thickness,
            draw_centers=draw_centers,
            center_radius=center_radius,
        )

        out_name = f"{save_idx:03d}_{img_path.stem}_crop{crop_size}.png"
        out_path = out_dir / out_name

        ok = cv2.imwrite(str(out_path), vis)
        if not ok:
            raise RuntimeError(f"Failed to save image: {out_path}")

        manifest_rows.append(
            {
                "saved_name": out_name,
                "source_image": img_path.name,
                "split": split,
                "crop_size": crop_size,
                "num_positive": int(len(pos_pts)),
                "num_negative": int(len(neg_pts)),
                "num_total": int(len(all_pts)),
            }
        )

    if write_manifest:
        manifest_path = out_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "saved_name",
                    "source_image",
                    "split",
                    "crop_size",
                    "num_positive",
                    "num_negative",
                    "num_total",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"Saved {len(selected_indices)} images to: {out_dir}")


if __name__ == "__main__":
    main()