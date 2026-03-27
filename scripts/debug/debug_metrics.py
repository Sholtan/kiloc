from kiloc.evaluation.metrics import compute_metrics

import torch
import numpy as np

# case 1
pred_pts = np.array([], dtype=np.float32)
gt_pts = np.array([], dtype=np.int64)
radius = 4.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# must be (0, 0, 0)
assert (0, 0, 0) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)



# case 2
pred_pts = np.array([], dtype=np.float32)
gt_pts = np.array([[10, 11], [23, 13]], dtype=np.int64)

radius = 4.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# must be (0, 0, 2)
#print("first test (must be (0, 0, 2)): ",(tp, fp, fn))
assert (0, 0, 2) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)



# case 3
pred_pts = np.array([[10, 11]], dtype=np.float32)
gt_pts = np.array([], dtype=np.int64)

radius = 4.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# must be (0, 1, 0)
assert (0, 1, 0) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)



# case 4
pred_pts = np.array([(100, 100)], dtype=np.float32)
gt_pts = np.array([(100, 100)], dtype=np.int64)

radius = 6.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# must be (1, 0, 0)
assert (1, 0, 0) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)


# case 5
pred_pts = np.array([(100, 100)], dtype=np.float32)
gt_pts = np.array([(200, 200)], dtype=np.int64)

radius = 6.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# must be (0, 1, 1)
assert (0, 1, 1) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)


# case 6
pred_pts = np.array([(100, 100), (102, 100)], dtype=np.float32)
gt_pts = np.array([(100, 100)], dtype=np.int64)

radius = 6.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# must be (1, 1, 0)
assert (1, 1, 0) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)


# case 7
pred_pts = np.array([(100, 100)], dtype=np.float32)
gt_pts = np.array([(100, 100), (100, 200)], dtype=np.int64)

radius = 6.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# must be (1, 0, 1)
assert (1, 0, 1) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)


# case 8
pred_pts = np.array([(5, 0), (0, 0)], dtype=np.float32)
gt_pts = np.array([(0, 0), (10, 0)], dtype=np.int64)

radius = 6.
tp, fp, fn = compute_metrics(pred_pts=pred_pts, gt_pts=gt_pts, radius=radius, )
# default matching is optimal now, so this should recover both matches
assert (2, 0, 0) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)

tp, fp, fn = compute_metrics(
    pred_pts=pred_pts,
    gt_pts=gt_pts,
    radius=radius,
    matching_mode="greedy",
)
assert (1, 1, 1) == (tp, fp, fn)

# Greedy is still order-dependent here:
# pred[0]=(5,0) claims gt[0]=(0,0) first, so pred[1]=(0,0) cannot recover gt[1]=(10,0).
# Optimal matching correctly returns (2, 0, 0) for the same points.

print("all cases passed")
