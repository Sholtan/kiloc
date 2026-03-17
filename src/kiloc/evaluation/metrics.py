import numpy as np
from numpy.typing import NDArray

from scipy.spatial.distance import cdist



def compute_metrics(
        pred_pts: NDArray,   # (N, 2) float32, image space, (x, y)
        gt_pts: NDArray,     # (M, 2) int64, image space, (x, y)
        radius: float,       # matching distance threshold in image pixels
) -> tuple[int, int, int]:
    '''
    Compute metrics for 1 sample

    Returns
    ----------
    tp, fp, fn
    '''
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0, 0, 0
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)   # all GT missed
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0  # all preds are FP

    dist = cdist(pred_pts, gt_pts)          # (N, M)
    matched_gt = set()
    tp = 0

    for i in range(len(pred_pts)):
        j = dist[i].argmin()
        if dist[i, j] <= radius and j not in matched_gt:
            tp += 1
            matched_gt.add(j)

    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def match_points(pred_pts, gt_pts, radius):
    """
    Returns:
        tp_pred  — pred coords that matched a GT point
        fp_pred  — pred coords with no GT match
        fn_gt    — GT coords not matched by any pred
    """
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return pred_pts, pred_pts, gt_pts
    if len(pred_pts) == 0:
        return pred_pts, pred_pts, gt_pts
    if len(gt_pts) == 0:
        return pred_pts[:0], pred_pts, gt_pts[:0]

    dist = cdist(pred_pts, gt_pts)
    matched_pred = set()
    matched_gt = set()

    for i in range(len(pred_pts)):
        j = dist[i].argmin()
        if dist[i, j] <= radius and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)

    tp_pred = pred_pts[sorted(matched_pred)]
    fp_pred = pred_pts[[i for i in range(len(pred_pts)) if i not in matched_pred]]
    fn_gt   = gt_pts[[j for j in range(len(gt_pts)) if j not in matched_gt]]
    return tp_pred, fp_pred, fn_gt