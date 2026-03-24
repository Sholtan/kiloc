from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import h5py

from kiloc.datasets.bcdata import BCDataDataset
from kiloc.utils.config import get_paths


def _dummy_target_transform(_: np.ndarray) -> None:
    # BCDataDataset requires a callable, but this script never calls __getitem__.
    return None

def _load_points(ann_path: Path) -> np.ndarray:
    with h5py.File(ann_path, "r") as f:
        return f["coordinates"][:]

def _add_box_to_mask(
    mask: np.ndarray,
    x: int,
    y: int,
    box_size: int,
) -> None:
    """
    Exclude a box of exactly `box_size x box_size` centered on (x, y),
    clipped to image boundaries.
    """
    h, w = mask.shape

    half_left = box_size // 2
    half_right = box_size - half_left

    x1 = max(0, x - half_left)
    x2 = min(w, x + half_right)
    y1 = max(0, y - half_left)
    y2 = min(h, y + half_right)

    mask[y1:y2, x1:x2] = False


def _histogram_median(hist: np.ndarray) -> float:
    """
    Exact median from a 256-bin histogram.
    Returns the same value np.median would return on uint8 data:
    for even counts, it averages the two middle order statistics.
    """
    total = int(hist.sum())
    if total == 0:
        raise ValueError("Cannot compute median from an empty histogram.")

    cumsum = np.cumsum(hist, dtype=np.uint64)

    k1 = (total - 1) // 2  # zero-based
    k2 = total // 2        # zero-based

    v1 = int(np.searchsorted(cumsum, k1 + 1, side="left"))
    v2 = int(np.searchsorted(cumsum, k2 + 1, side="left"))

    return 0.5 * (v1 + v2)


def _build_dataset(root: Path, split: str) -> BCDataDataset:
    return BCDataDataset(
        root=root,
        split=split,
        target_transform=_dummy_target_transform,  # only needed for constructor
        joint_transform=None,
        input_normalization=None,
    )


def compute_background_median(
    data_root: Path,
    split: str = "train",
    exclude_box_size: int = 192,
    progress_every: int = 50,
) -> dict:
    if exclude_box_size <= 0:
        raise ValueError("exclude_box_size must be positive.")

    dataset = _build_dataset(root=data_root, split=split)

    # Exact per-channel median for uint8 images via histograms.
    hist_r = np.zeros(256, dtype=np.uint64)
    hist_g = np.zeros(256, dtype=np.uint64)
    hist_b = np.zeros(256, dtype=np.uint64)

    total_pixels_all = 0
    kept_pixels_all = 0
    total_points_all = 0
    skipped_images = 0

    for idx, (img_path, pos_ann_path, neg_ann_path) in enumerate(dataset.samples, start=1):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        pos_pts = _load_points(pos_ann_path)
        neg_pts = _load_points(neg_ann_path)

        if pos_pts.size == 0 and neg_pts.size == 0:
            all_pts = np.empty((0, 2), dtype=np.int64)
        elif pos_pts.size == 0:
            all_pts = np.asarray(neg_pts, dtype=np.int64)
        elif neg_pts.size == 0:
            all_pts = np.asarray(pos_pts, dtype=np.int64)
        else:
            all_pts = np.concatenate([pos_pts, neg_pts], axis=0).astype(np.int64, copy=False)

        total_points_all += int(all_pts.shape[0])

        mask = np.ones((h, w), dtype=bool)

        for pt in all_pts:
            x = int(round(float(pt[0])))
            y = int(round(float(pt[1])))
            _add_box_to_mask(mask=mask, x=x, y=y, box_size=exclude_box_size)

        valid_pixels = img_rgb[mask]  # shape: (N, 3), dtype uint8

        total_pixels_all += h * w
        kept_pixels_all += int(valid_pixels.shape[0])

        if valid_pixels.shape[0] == 0:
            skipped_images += 1
            continue

        hist_r += np.bincount(valid_pixels[:, 0], minlength=256).astype(np.uint64)
        hist_g += np.bincount(valid_pixels[:, 1], minlength=256).astype(np.uint64)
        hist_b += np.bincount(valid_pixels[:, 2], minlength=256).astype(np.uint64)

        if progress_every > 0 and (idx % progress_every == 0 or idx == len(dataset)):
            kept_frac = kept_pixels_all / max(total_pixels_all, 1)
            print(
                f"[{idx:4d}/{len(dataset):4d}] "
                f"kept_pixels={kept_pixels_all} "
                f"kept_fraction={kept_frac:.4f}"
            )

    median_r = _histogram_median(hist_r)
    median_g = _histogram_median(hist_g)
    median_b = _histogram_median(hist_b)

    median_rgb_uint8 = [median_r, median_g, median_b]
    median_rgb_uint8_rounded = [int(round(v)) for v in median_rgb_uint8]
    median_rgb_float = [v / 255.0 for v in median_rgb_uint8]

    result = {
        "split": split,
        "exclude_box_size": exclude_box_size,
        "num_images": len(dataset),
        "num_points_total": total_points_all,
        "total_pixels": int(total_pixels_all),
        "kept_pixels": int(kept_pixels_all),
        "kept_fraction": float(kept_pixels_all / max(total_pixels_all, 1)),
        "skipped_images_with_no_valid_pixels": skipped_images,
        "median_rgb_uint8": median_rgb_uint8,
        "median_rgb_uint8_rounded": median_rgb_uint8_rounded,
        "median_rgb_float_0_1": median_rgb_float,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-channel median background color on BCData by masking out "
            "192x192 neighborhoods (or another specified size) around all annotated cells."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="h200",
        choices=["hpvictus", "collab", "h200"],
        help="Device key used by kiloc.utils.config.get_paths().",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use for the estimate.",
    )
    parser.add_argument(
        "--exclude-box-size",
        type=int,
        default=192,
        help="Side length of the exclusion box centered at each annotated point.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/bcdata_background_median.json"),
        help="Where to save the computed statistics as JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N images. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root, _ = get_paths(device=args.device)

    result = compute_background_median(
        data_root=data_root,
        split=args.split,
        exclude_box_size=args.exclude_box_size,
        progress_every=args.progress_every,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print()
    print(
        "Note: values are reported in RGB order. "
        "If you pad before cv2.cvtColor(..., cv2.COLOR_BGR2RGB), reverse them to BGR."
    )


if __name__ == "__main__":
    main()