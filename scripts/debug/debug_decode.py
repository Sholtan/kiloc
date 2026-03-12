from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.utils.debug import print_info
from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.utils.config import get_paths
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.visualization.plots import plot_points
import torch
import numpy as np
from torch.utils.data import DataLoader

# First test, 1 cell in 1 channel
hm = torch.zeros(1, 2, 160, 160)   # B = 1
# place a peak at (x=40, y=30) on channel 0
x_true, y_true = 30, 40
hm[0, 0, y_true, x_true] = 1.0
pos_pts, neg_pts = heatmaps_to_points_batch(
    hm, kernel_size=3, threshold=0.5, output_hw=(640, 640))
# expect pos_pts[0] ≈ [[160, 120]] (scaled by 4)
# expect neg_pts[0] shape (0, 2)
x_err = np.abs(pos_pts[0][0, 0] - 4. * x_true)
y_err = np.abs(pos_pts[0][0, 1] - 4. * y_true)
assert x_err < 0.1, f"first test, error in x: {x_err}"
assert y_err < 0.1, f"first test, error in y: {y_err}"


# Second test
hm_empty = torch.zeros(1, 2, 160, 160)
pos_pts, neg_pts = heatmaps_to_points_batch(
    hm_empty, kernel_size=3, threshold=0.5, output_hw=(640, 640))
assert pos_pts[0].shape == (0, 2), f"Second test, failed on empty heatmap"
assert neg_pts[0].shape == (0, 2), f"Second test, failed on empty heatmap"

# Third test
rootdir, _ = get_paths(device='hpvictus')
heatmap_gen = LocHeatmap(out_hw=(160, 160), in_hw=(640, 640))
bc_dataset = BCDataDataset(rootdir, 'train', target_transform=heatmap_gen)
dataloader = DataLoader(dataset=bc_dataset, batch_size=1,
                        shuffle=False, collate_fn=collate_fn)
imgs, heatmaps, pos_gt, neg_gt = next(iter(dataloader))
# pass ground truth heatmaps (not model output) through decoder
pos_pred, neg_pred = heatmaps_to_points_batch(
    heatmaps, kernel_size=3, threshold=0.5, output_hw=(640, 640), refine=True)
# compare pos_pred[0] visually against pos_gt[0]

plot_points(image=imgs, points_gt=pos_gt, pos_pred=pos_pred)
