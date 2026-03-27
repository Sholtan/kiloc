from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kiloc.oof.relations import load_gt_by_image_from_image_paths
from kiloc.three_classifier.datasets import CLASS_ID_TO_NAME, read_image_rgb


@dataclass(frozen=True)
class RankedCategorySpec:
    slug: str
    description: str
    true_label: int
    score_key: str
    color_bgr: tuple[int, int, int]
    selected_marker: str = "cross"


RANKED_CATEGORY_SPECS: dict[str, RankedCategorySpec] = {
    "mined_to_ann_pos": RankedCategorySpec(
        slug="mined_to_ann_pos",
        description="true mined crops with highest ann_pos score",
        true_label=2,
        score_key="prob_class_0",
        color_bgr=(30, 144, 255),
        selected_marker="circle",
    ),
    "mined_to_ann_neg": RankedCategorySpec(
        slug="mined_to_ann_neg",
        description="true mined crops with highest ann_neg score",
        true_label=2,
        score_key="prob_class_1",
        color_bgr=(0, 165, 255),
        selected_marker="circle",
    ),
    "ann_neg_to_mined": RankedCategorySpec(
        slug="ann_neg_to_mined",
        description="true ann_neg crops with highest mined score",
        true_label=1,
        score_key="prob_class_2",
        color_bgr=(60, 179, 113),
        selected_marker="circle",
    ),
    "ann_pos_to_mined": RankedCategorySpec(
        slug="ann_pos_to_mined",
        description="true ann_pos crops with highest mined score",
        true_label=0,
        score_key="prob_class_2",
        color_bgr=(106, 90, 205),
        selected_marker="circle",
    ),
}

_SHORT_CLASS_NAMES = {
    0: "ann_pos",
    1: "ann_neg",
    2: "mined",
}


def load_prediction_rows(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = dict(raw_row)
            for key in ("dataset_index", "true_label", "pred_label", "crop_size", "is_correct"):
                if row.get(key, "") == "":
                    row[key] = 0
                else:
                    row[key] = int(float(row[key]))
            for key in (
                "x",
                "y",
                "prob_class_0",
                "prob_class_1",
                "prob_class_2",
                "entropy",
                "margin",
                "source_score",
            ):
                value = row.get(key, "")
                row[key] = float(value) if value not in ("", None) else float("nan")
            rows.append(row)
    return rows


def rank_rows_for_category(
    *,
    rows: list[dict[str, object]],
    category: RankedCategorySpec,
    top_k: int,
) -> list[dict[str, object]]:
    filtered = [
        row
        for row in rows
        if int(row["true_label"]) == category.true_label
    ]
    ranked = sorted(
        filtered,
        key=lambda row: (
            float(row[category.score_key]),
            float(row.get("entropy", 0.0)),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def _resolve_image_path(*, data_root: str | Path, image_path: str) -> Path:
    return Path(data_root) / image_path


def _clipped_box(
    *,
    center_x: float,
    center_y: float,
    box_size: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    half = box_size // 2
    cx = int(round(center_x))
    cy = int(round(center_y))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(width - 1, cx + half)
    y1 = min(height - 1, cy + half)
    return x0, y0, x1, y1


def draw_full_image_example(
    *,
    image_rgb: np.ndarray,
    row: dict[str, object],
    category: RankedCategorySpec,
    box_size: int,
    gt_pos_points: np.ndarray,
    gt_neg_points: np.ndarray,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_height, width = image_bgr.shape[:2]
    gt_pos_color = (220, 60, 220)
    gt_neg_color = (80, 220, 80)
    selected_true_label = int(row["true_label"])
    selected_xy = np.array([float(row["x"]), float(row["y"])], dtype=np.float32)

    vis_gt_pos_points = np.asarray(gt_pos_points, dtype=np.float32).reshape(-1, 2)
    vis_gt_neg_points = np.asarray(gt_neg_points, dtype=np.float32).reshape(-1, 2)
    if category.selected_marker == "circle":
        if selected_true_label == 0 and len(vis_gt_pos_points) > 0:
            distances = np.linalg.norm(vis_gt_pos_points - selected_xy[None, :], axis=1)
            vis_gt_pos_points = np.delete(vis_gt_pos_points, int(distances.argmin()), axis=0)
        elif selected_true_label == 1 and len(vis_gt_neg_points) > 0:
            distances = np.linalg.norm(vis_gt_neg_points - selected_xy[None, :], axis=1)
            vis_gt_neg_points = np.delete(vis_gt_neg_points, int(distances.argmin()), axis=0)

    for points, point_color in ((vis_gt_pos_points, gt_pos_color), (vis_gt_neg_points, gt_neg_color)):
        for point in np.asarray(points, dtype=np.float32).reshape(-1, 2):
            px = int(round(float(point[0])))
            py = int(round(float(point[1])))
            cv2.circle(image_bgr, (px, py), 4, point_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image_bgr, (px, py), 6, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    x0, y0, x1, y1 = _clipped_box(
        center_x=float(row["x"]),
        center_y=float(row["y"]),
        box_size=box_size,
        width=width,
        height=image_height,
    )

    color = category.color_bgr
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color, thickness=3)

    cx = int(round(float(row["x"])))
    cy = int(round(float(row["y"])))
    if category.selected_marker == "cross":
        cross_radius = max(4, box_size // 16)
        cv2.line(
            image_bgr,
            (max(0, cx - cross_radius), cy),
            (min(width - 1, cx + cross_radius), cy),
            color,
            2,
        )
        cv2.line(
            image_bgr,
            (cx, max(0, cy - cross_radius)),
            (cx, min(image_height - 1, cy + cross_radius)),
            color,
            2,
        )
        marker_legend = "selected marker=cross"
    elif category.selected_marker == "circle":
        cv2.circle(
            image_bgr,
            (cx, cy),
            50,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        marker_legend = "selected marker=circle r=50"
    else:
        raise ValueError(f"Unsupported selected_marker={category.selected_marker!r}")

    true_label = int(row["true_label"])
    pred_label = int(row["pred_label"])
    overlay_lines = [
        category.description,
        (
            f"true={_SHORT_CLASS_NAMES[true_label]} "
            f"pred={_SHORT_CLASS_NAMES[pred_label]} "
            f"sample_id={row['sample_id']}"
        ),
        (
            f"p_ann_pos={float(row['prob_class_0']):.3f} "
            f"p_ann_neg={float(row['prob_class_1']):.3f} "
            f"p_mined={float(row['prob_class_2']):.3f}"
        ),
        (
            f"image_id={row['image_id']} center=({float(row['x']):.1f}, {float(row['y']):.1f}) "
            f"GT_pos={len(vis_gt_pos_points)} GT_neg={len(vis_gt_neg_points)}"
        ),
        f"legend: box={box_size} crop window, {marker_legend}, magenta=GT ann_pos, green=GT ann_neg",
    ]

    text_scale = 0.5
    text_thickness = 1
    padding = 10
    line_gap = 6
    line_heights = []
    for line in overlay_lines:
        (_text_width, text_height), baseline = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        line_heights.append(text_height + baseline)

    header_height = sum(line_heights) + line_gap * (len(overlay_lines) - 1) + 2 * padding
    canvas = np.full((image_height + header_height, width, 3), 245, dtype=np.uint8)
    header = canvas[:header_height]
    header[:] = (28, 28, 28)
    canvas[header_height:] = image_bgr

    cv2.rectangle(
        header,
        (0, 0),
        (width - 1, header_height - 1),
        color,
        thickness=2,
    )

    y_cursor = padding + line_heights[0]
    for index, line in enumerate(overlay_lines):
        cv2.putText(
            header,
            line,
            (padding + 4, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
        if index + 1 < len(overlay_lines):
            y_cursor += line_heights[index + 1] + line_gap
    return canvas


def save_ranked_full_image_examples(
    *,
    rows: list[dict[str, object]],
    data_root: str | Path,
    out_dir: str | Path,
    category_slugs: list[str],
    top_k: int,
    box_size: int,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, np.ndarray] = {}
    gt_by_image = load_gt_by_image_from_image_paths(
        data_root=data_root,
        image_paths=sorted({str(row["image_path"]) for row in rows}),
    )

    summary_categories: dict[str, dict[str, object]] = {}
    for slug in category_slugs:
        if slug not in RANKED_CATEGORY_SPECS:
            raise ValueError(
                f"Unknown category {slug}. Expected one of {sorted(RANKED_CATEGORY_SPECS)}"
            )
        spec = RANKED_CATEGORY_SPECS[slug]
        ranked_rows = rank_rows_for_category(rows=rows, category=spec, top_k=top_k)

        category_dir = out_dir / slug
        category_dir.mkdir(parents=True, exist_ok=True)
        ranked_csv = category_dir / f"{slug}_top.csv"
        _write_csv_rows(rows=ranked_rows, out_csv=ranked_csv)

        for rank_index, row in enumerate(ranked_rows, start=1):
            image_rgb = read_image_rgb(
                _resolve_image_path(data_root=data_root, image_path=str(row["image_path"])),
                cache=image_cache,
            )
            rendered = draw_full_image_example(
                image_rgb=image_rgb,
                row=row,
                category=spec,
                box_size=box_size,
                gt_pos_points=np.asarray(gt_by_image[str(row["image_id"])]["pos_points"], dtype=np.float32),
                gt_neg_points=np.asarray(gt_by_image[str(row["image_id"])]["neg_points"], dtype=np.float32),
            )
            filename = (
                f"{rank_index:04d}_"
                f"score{float(row[spec.score_key]):.4f}_"
                f"{str(row['image_id'])}_"
                f"{str(row['sample_id'])}.png"
            )
            cv2.imwrite(str(category_dir / filename), rendered)

        summary_categories[slug] = {
            "description": spec.description,
            "score_key": spec.score_key,
            "true_label": spec.true_label,
            "true_label_name": CLASS_ID_TO_NAME[spec.true_label],
            "selected_marker": spec.selected_marker,
            "top_k_saved": len(ranked_rows),
            "csv_path": ranked_csv.as_posix(),
            "image_dir": category_dir.as_posix(),
        }

    summary = {
        "top_k": int(top_k),
        "box_size": int(box_size),
        "categories": summary_categories,
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _write_csv_rows(*, rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
