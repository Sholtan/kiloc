from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def _sample_channel_bilinear(
    channel: torch.Tensor,
    points_xy: np.ndarray,
    output_hw: tuple[int, int],
) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if channel.ndim != 2:
        raise ValueError(f"Expected 2D channel tensor, got shape {tuple(channel.shape)}")

    dst_h, dst_w = output_hw
    src_h, src_w = int(channel.shape[0]), int(channel.shape[1])

    x = points_xy[:, 0].astype(np.float32) * (src_w / float(dst_w))
    y = points_xy[:, 1].astype(np.float32) * (src_h / float(dst_h))

    x = np.clip(x, 0.0, max(src_w - 1, 0))
    y = np.clip(y, 0.0, max(src_h - 1, 0))

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)

    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)

    channel_np = channel.detach().cpu().numpy().astype(np.float32, copy=False)

    v00 = channel_np[y0, x0]
    v01 = channel_np[y0, x1]
    v10 = channel_np[y1, x0]
    v11 = channel_np[y1, x1]

    values = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )
    return values.astype(np.float32, copy=False)


def build_raw_prediction_rows(
    *,
    fold: int,
    image_id: str,
    image_path: str,
    points_xy: np.ndarray,
    pred_class: str,
    heatmap_pos: torch.Tensor,
    heatmap_neg: torch.Tensor,
    output_hw: tuple[int, int],
    threshold: tuple[float, float],
    model_checkpoint: str,
) -> list[dict[str, str | int | float]]:
    if pred_class not in {"pos", "neg"}:
        raise ValueError(f"pred_class must be 'pos' or 'neg', got {pred_class!r}")

    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points_xy.size == 0:
        return []

    scores_pos = _sample_channel_bilinear(heatmap_pos, points_xy, output_hw=output_hw)
    scores_neg = _sample_channel_bilinear(heatmap_neg, points_xy, output_hw=output_hw)
    scores = scores_pos if pred_class == "pos" else scores_neg
    decode_threshold = float(threshold[0] if pred_class == "pos" else threshold[1])

    order = np.argsort(-scores)
    rows: list[dict[str, str | int | float]] = []
    for rank, idx in enumerate(order):
        rows.append(
            {
                "fold": fold,
                "image_id": image_id,
                "image_path": image_path,
                "pred_id": f"{image_id}_{pred_class}_{rank:04d}",
                "pred_class": pred_class,
                "x": float(points_xy[idx, 0]),
                "y": float(points_xy[idx, 1]),
                "score": float(scores[idx]),
                "score_pos": float(scores_pos[idx]),
                "score_neg": float(scores_neg[idx]),
                "decode_threshold": decode_threshold,
                "model_checkpoint": model_checkpoint,
                "raw_rank_in_image": rank,
            }
        )

    return rows


def write_raw_prediction_csv(
    rows: list[dict[str, str | int | float]],
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
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
