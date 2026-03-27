from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _validate_matching_mode(matching_mode: str) -> str:
    if matching_mode not in {"greedy", "optimal"}:
        raise ValueError(
            f"matching_mode must be one of {{'greedy', 'optimal'}}, got {matching_mode!r}"
        )
    return matching_mode


def _empty_index_array() -> NDArray[np.int64]:
    return np.empty((0,), dtype=np.int64)


def _greedy_match_indices(
    dist: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    matched_pred: list[int] = []
    matched_gt: list[int] = []
    used_gt: set[int] = set()

    for pred_idx in range(dist.shape[0]):
        gt_idx = int(dist[pred_idx].argmin())
        if dist[pred_idx, gt_idx] <= radius and gt_idx not in used_gt:
            matched_pred.append(pred_idx)
            matched_gt.append(gt_idx)
            used_gt.add(gt_idx)

    return np.asarray(matched_pred, dtype=np.int64), np.asarray(matched_gt, dtype=np.int64)


def _optimal_match_indices(
    dist: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    valid = dist <= radius
    if not np.any(valid):
        return _empty_index_array(), _empty_index_array()

    max_valid_cost = float(dist[valid].max()) if np.any(valid) else 0.0
    max_pairs = min(dist.shape[0], dist.shape[1])
    invalid_cost = (max_valid_cost + 1.0) * (max_pairs + 1)

    cost = dist.copy()
    cost[~valid] = invalid_cost

    pred_indices, gt_indices = linear_sum_assignment(cost)
    keep = valid[pred_indices, gt_indices]
    return (
        pred_indices[keep].astype(np.int64, copy=False),
        gt_indices[keep].astype(np.int64, copy=False),
    )


def _match_indices(
    pred_pts: NDArray,
    gt_pts: NDArray,
    radius: float,
    matching_mode: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    pred_pts = np.asarray(pred_pts, dtype=np.float32).reshape(-1, 2)
    gt_pts = np.asarray(gt_pts, dtype=np.float32).reshape(-1, 2)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return _empty_index_array(), _empty_index_array()

    dist = cdist(pred_pts, gt_pts)
    if matching_mode == "greedy":
        return _greedy_match_indices(dist, radius)
    return _optimal_match_indices(dist, radius)


def match_indices(
    pred_pts: NDArray,
    gt_pts: NDArray,
    radius: float,
    matching_mode: str = "optimal",
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    matching_mode = _validate_matching_mode(matching_mode)
    return _match_indices(
        pred_pts=pred_pts,
        gt_pts=gt_pts,
        radius=radius,
        matching_mode=matching_mode,
    )


def compute_metrics(
    pred_pts: NDArray,   # (N, 2) float32, image space, (x, y)
    gt_pts: NDArray,     # (M, 2) int64, image space, (x, y)
    radius: float,       # matching distance threshold in image pixels
    matching_mode: str = "optimal",
) -> tuple[int, int, int]:
    """
    Compute metrics for 1 sample.

    Returns
    ----------
    tp, fp, fn
    """
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0, 0, 0
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0

    matching_mode = _validate_matching_mode(matching_mode)
    matched_pred, _ = match_indices(
        pred_pts=pred_pts,
        gt_pts=gt_pts,
        radius=radius,
        matching_mode=matching_mode,
    )
    tp = int(len(matched_pred))
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def match_points(
    pred_pts: NDArray,
    gt_pts: NDArray,
    radius: float,
    matching_mode: str = "optimal",
):
    """
    Returns:
        tp_pred  - pred coords that matched a GT point
        fp_pred  - pred coords with no GT match
        fn_gt    - GT coords not matched by any pred
    """
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return pred_pts, pred_pts, gt_pts
    if len(pred_pts) == 0:
        return pred_pts, pred_pts, gt_pts
    if len(gt_pts) == 0:
        return pred_pts[:0], pred_pts, gt_pts[:0]

    matching_mode = _validate_matching_mode(matching_mode)
    matched_pred, matched_gt = match_indices(
        pred_pts=pred_pts,
        gt_pts=gt_pts,
        radius=radius,
        matching_mode=matching_mode,
    )

    matched_pred_set = set(int(idx) for idx in matched_pred.tolist())
    matched_gt_set = set(int(idx) for idx in matched_gt.tolist())

    tp_pred = pred_pts[matched_pred]
    fp_pred = pred_pts[[i for i in range(len(pred_pts)) if i not in matched_pred_set]]
    fn_gt = gt_pts[[j for j in range(len(gt_pts)) if j not in matched_gt_set]]
    return tp_pred, fp_pred, fn_gt
