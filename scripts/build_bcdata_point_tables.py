from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kiloc.utils.config import get_paths


COLUMNS = [
    "point_id",
    "image_path",
    "image_id",
    "x",
    "y",
    "label",
]

SPLITS = ("train", "validation", "test")
LABEL_TO_INT = {
    "positive": 1,
    "negative": 0,
}

def _sorted_png_paths(image_dir: Path) -> list[Path]:
    paths = list(image_dir.glob("*.png"))

    def sort_key(p: Path) -> tuple[int, Any]:
        # Match deterministic order of sorted(...) used in bcdata.py,
        # but handle numeric stems more naturally if possible.
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(paths, key=sort_key)

def load_points(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        coords = f["coordinates"][:]

    coords = np.asarray(coords)

    # Empty annotation file: no cells of this class in this image
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Expected coordinates with shape (N, 2) in {h5_path}, got {coords.shape}"
        )

    return coords


def build_rows_for_split(
    data_root: Path,
    split: str,
    start_point_id: int,
) -> tuple[list[dict[str, Any]], int]:
    image_dir = data_root / "images" / split
    pos_ann_dir = data_root / "annotations" / split / "positive"
    neg_ann_dir = data_root / "annotations" / split / "negative"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not pos_ann_dir.exists():
        raise FileNotFoundError(f"Missing positive annotation directory: {pos_ann_dir}")
    if not neg_ann_dir.exists():
        raise FileNotFoundError(f"Missing negative annotation directory: {neg_ann_dir}")

    image_paths = _sorted_png_paths(image_dir)
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    rows: list[dict[str, Any]] = []
    point_id = start_point_id

    for img_path in image_paths:
        stem = img_path.stem
        image_id: int | str = int(stem) if stem.isdigit() else stem
        rel_img_path = img_path.relative_to(data_root).as_posix()

        pos_h5 = pos_ann_dir / f"{stem}.h5"
        neg_h5 = neg_ann_dir / f"{stem}.h5"

        if not pos_h5.exists():
            raise FileNotFoundError(f"Missing annotation file: {pos_h5}")
        if not neg_h5.exists():
            raise FileNotFoundError(f"Missing annotation file: {neg_h5}")

        pos_points = load_points(pos_h5)
        neg_points = load_points(neg_h5)

        for label_name, points in (("positive", pos_points), ("negative", neg_points)):
            label = LABEL_TO_INT[label_name]

            for xy in points:
                x = int(xy[0])
                y = int(xy[1])

                rows.append(
                    {
                        "point_id": point_id,
                        "image_path": rel_img_path,
                        "image_id": image_id,
                        "x": x,
                        "y": y,
                        "label": label,
                    }
                )
                point_id += 1

    return rows, point_id


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Change these two paths as needed.
    data_root, _ = get_paths(device='h200')
    out_dir = Path("./bcdata_point_tables")

    next_point_id = 0
    for split in SPLITS:
        rows, next_point_id = build_rows_for_split(
            data_root=data_root,
            split=split,
            start_point_id=next_point_id,
        )

        out_path = out_dir / f"{split}_points.csv"
        write_csv(rows, out_path)
        n_pos = sum(r["label"] == 1 for r in rows)
        n_neg = sum(r["label"] == 0 for r in rows)
        n_images = len({r["image_id"] for r in rows})

        print(
            f"{split:10s} -> {out_path} | "
            f"rows={len(rows)} images={n_images} pos={n_pos} neg={n_neg}"
        )


if __name__ == "__main__":
    main()