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