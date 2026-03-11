from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.utils.debug import print_info
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import cv2

from kiloc.utils.config import get_paths


dataroot, _ = get_paths('hpvictus')


img_path = dataroot / "images" / "train" / "0.png"
ann_pos_path = dataroot / "annotations" / "train" / "positive" / "0.h5"
ann_neg_path = dataroot / "annotations" / "train" / "negative" / "0.h5"

heatmap_generator = LocHeatmap(out_hw=(160, 160), in_hw=(
    640, 640), sigma=3.0, dtype=torch.float32)


with h5py.File(ann_pos_path, 'r') as f:
    dset = f["coordinates"]
    assert isinstance(dset, h5py.Dataset)
    coords_pos = dset[:]

with h5py.File(ann_neg_path, 'r') as f:
    dset = f["coordinates"]
    assert isinstance(dset, h5py.Dataset)
    coords_neg = dset[:]

hm_pos = heatmap_generator(coords_pos)
hm_neg = heatmap_generator(coords_neg)


img = cv2.imread(
    img_path)    # pixel value range: 0 - 255, BGR, (H, W, C)
if img is None:
    raise RuntimeError(f"Failed to read image at {img_path}")

img_scaled = cv2.cvtColor(
    img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

img_tensor: torch.Tensor = torch.from_numpy(
    img_scaled).permute(2, 0, 1).contiguous()


fig, ax = plt.subplots(1, 3, figsize=(12, 10))

ax[0].imshow(img_scaled)
ax[0].set_title("original")
ax[1].imshow(hm_pos[0])
ax[1].set_title("positive's heatmap")

ax[2].imshow(hm_neg[0])
ax[2].set_title("negative's heatmap")

# plt.show()
plt.savefig("./images/heatmap.png")
