from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


DEFAULT_VISUAL_FILTER_REASONS = (
    "kept_for_mining",
    "below_score_threshold",
    "near_gt_any",
    "interclass_conflict",
    "sameclass_duplicate",
)

GT_POS_COLOR = "#3ddc84"
GT_NEG_COLOR = "#4db6ff"
PRED_POS_COLOR = "#ffd54f"
PRED_NEG_COLOR = "#ff8a65"
SELECTED_COLOR = "#ff4dd2"


def _rows_to_points(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[row["x"], row["y"]] for row in rows], dtype=np.float32)


def _load_image_rgb(image_path: Path) -> np.ndarray:
    import cv2

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image_rgb


def _scatter_points(
    ax: plt.Axes,
    points_xy: np.ndarray,
    *,
    label: str,
    edge_color: str,
    face_color: str = "none",
    marker: str = "o",
    size: float = 52.0,
    linewidths: float = 1.6,
    alpha: float = 1.0,
) -> None:
    if len(points_xy) == 0:
        return

    ax.scatter(
        points_xy[:, 0],
        points_xy[:, 1],
        s=size,
        marker=marker,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=linewidths,
        alpha=alpha,
        label=label,
    )


def _dedupe_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return

    deduped: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        deduped[label] = handle
    ax.legend(deduped.values(), deduped.keys(), loc="lower right", fontsize=8, framealpha=0.9)


def _image_sort_key(image_id: str) -> tuple[int, str]:
    if image_id.isdigit():
        return (0, f"{int(image_id):08d}")
    return (1, image_id)


def _plot_reason_image(
    *,
    image_rgb: np.ndarray,
    image_rows: list[dict[str, Any]],
    image_gt: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    reason: str,
) -> plt.Figure:
    fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(16, 8))

    pred_pos_rows = [row for row in image_rows if row["pred_class"] == "pos"]
    pred_neg_rows = [row for row in image_rows if row["pred_class"] == "neg"]
    pred_pos_points = _rows_to_points(pred_pos_rows)
    pred_neg_points = _rows_to_points(pred_neg_rows)
    selected_points = _rows_to_points(selected_rows)
    selected_pos_count = sum(1 for row in selected_rows if row["pred_class"] == "pos")
    selected_neg_count = sum(1 for row in selected_rows if row["pred_class"] == "neg")
    gt_pos = np.asarray(image_gt["pos_points"], dtype=np.float32).reshape(-1, 2)
    gt_neg = np.asarray(image_gt["neg_points"], dtype=np.float32).reshape(-1, 2)

    ax_pred.imshow(image_rgb)
    _scatter_points(
        ax_pred,
        pred_pos_points,
        label="Pred pos",
        edge_color=PRED_POS_COLOR,
        face_color=PRED_POS_COLOR,
        marker="x",
        size=26.0,
        linewidths=1.0,
        alpha=0.85,
    )
    _scatter_points(
        ax_pred,
        pred_neg_points,
        label="Pred neg",
        edge_color=PRED_NEG_COLOR,
        face_color=PRED_NEG_COLOR,
        marker="x",
        size=26.0,
        linewidths=1.0,
        alpha=0.85,
    )
    _scatter_points(
        ax_pred,
        selected_points,
        label=f"Selected: {reason}",
        edge_color=SELECTED_COLOR,
        face_color="none",
        marker="o",
        size=120.0,
        linewidths=2.0,
    )
    ax_pred.axis("off")
    _dedupe_legend(ax_pred)
    ax_pred.set_title(
        f"Predictions | reason={reason} | selected={len(selected_rows)}"
    )

    ax_gt.imshow(image_rgb)
    _scatter_points(ax_gt, gt_pos, label="GT pos", edge_color=GT_POS_COLOR, size=62.0)
    _scatter_points(ax_gt, gt_neg, label="GT neg", edge_color=GT_NEG_COLOR, size=62.0)
    ax_gt.axis("off")
    _dedupe_legend(ax_gt)
    ax_gt.set_title(f"Annotations | pos={len(gt_pos)} neg={len(gt_neg)}")

    fig.suptitle(
        " | ".join(
            [
                f"fold={selected_rows[0]['fold']}",
                f"image={selected_rows[0]['image_id']}",
                f"reason={reason}",
                f"selected_total={len(selected_rows)}",
                f"selected_pos={selected_pos_count}",
                f"selected_neg={selected_neg_count}",
                f"pred_pos_total={len(pred_pos_rows)}",
                f"pred_neg_total={len(pred_neg_rows)}",
            ]
        ),
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def visualize_relation_candidates(
    *,
    relation_rows: list[dict[str, Any]],
    gt_by_image: dict[str, dict[str, Any]],
    data_root: str | Path,
    output_dir: str | Path,
    filter_reasons: tuple[str, ...] | list[str],
    max_images_per_reason: int | None = None,
    dpi: int = 150,
) -> dict[str, int]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in relation_rows:
        rows_by_image.setdefault(row["image_id"], []).append(row)

    image_cache: dict[str, np.ndarray] = {}
    saved_counts: dict[str, int] = {}

    for reason in filter_reasons:
        reason_dir = output_dir / reason
        reason_dir.mkdir(parents=True, exist_ok=True)

        selected_rows_by_image: dict[str, list[dict[str, Any]]] = {}
        for row in relation_rows:
            if row["filter_reason"] != reason:
                continue
            selected_rows_by_image.setdefault(row["image_id"], []).append(row)

        ordered_images = sorted(selected_rows_by_image.items(), key=lambda item: _image_sort_key(item[0]))
        if max_images_per_reason is not None:
            ordered_images = ordered_images[:max_images_per_reason]

        print(f"[visualize] reason={reason} images={len(ordered_images)} out_dir={reason_dir}")
        saved_counts[reason] = 0
        for index, (image_id, selected_rows) in enumerate(
            tqdm(ordered_images, desc=f"{reason}", unit="image")
        ):
            image_path = data_root / selected_rows[0]["image_path"]
            if image_id not in image_cache:
                image_cache[image_id] = _load_image_rgb(image_path)

            fig = _plot_reason_image(
                image_rgb=image_cache[image_id],
                image_rows=rows_by_image[image_id],
                image_gt=gt_by_image[image_id],
                selected_rows=selected_rows,
                reason=reason,
            )
            filename = f"{index:05d}_image_{image_id}_selected_{len(selected_rows):03d}.png"
            fig.savefig(reason_dir / filename, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_counts[reason] += 1

    return saved_counts
