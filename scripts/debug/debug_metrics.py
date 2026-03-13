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
# must be (1, 1, 1)
assert (1, 1, 1) == (tp, fp, fn)
assert tp + fp == len(pred_pts)
assert tp + fn == len(gt_pts)

# The optimal assignment for case 8 would be (2, 0, 0):
# pred=(0,0) → gt=(0,0) (dist 0)
# pred=(5,0) → gt=(10,0) (dist 5, within radius 6)
# But the greedy implementation matches pred=(5,0) first (it's pred[0]), picks gt=(0,0) (dist 5, ties broken by argmin returning first index), then pred=(0,0) finds gt=(0,0) already matched and gt=(10,0) at dist 10 > 6.

# So (1, 1, 1) is correct for the greedy implementation, but the comment should say so explicitly. Otherwise when someone sees this result during debugging they'll think it's a bug. Add a note like:


# # greedy order-dependent: pred[0]=(5,0) claims gt[0]=(0,0) before pred[1]=(0,0) can
# This is a known limitation of greedy matching — it's the accepted design per decisions.md, but case 8 is exactly the failure mode worth documenting.

print("all cases passed")
