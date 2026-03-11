from torch.utils.data import DataLoader
from kiloc import BCDataDataset

from pathlib import Path

import torch

from kiloc.utils.config import get_paths

from kiloc.target_generation.heatmaps_batchs import Locheatmaps_batch
from kiloc.visualization.plots import save_image_heatmaps_batchs


print("start debug dataset")
# root_dir = Path(
#    "/home/yeldos/Istanbul/obsidian_vault/work_AITU/IHC/DATASETS/BCData")

root_dir, _ = get_paths("hpvictus")

heatmaps_batch_generator = Locheatmaps_batch(out_hw=(160, 160), in_hw=(
    640, 640), sigma=3.0, dtype=torch.float32)


dataset = BCDataDataset(root=root_dir, split='train', target_transform=heatmaps_batch_generator,
                        image_transform=None, joint_transform=None)

print(f"len(dataset): {len(dataset)}")


img_batch, heatmaps_batch, pos_pts_tuple, neg_pts_tuple = dataset[0]
assert img_batch.shape == (3, 640, 640), img_batch.shape
assert img_batch.dtype == torch.float32, img_batch.dtype
assert heatmaps_batch.shape == (2, 160, 160), heatmaps_batch.shape
assert heatmaps_batch.dtype == torch.float32, heatmaps_batch.dtype
assert img_batch.min() >= 0.0 and img_batch.max() <= 1.0
assert heatmaps_batch.min() >= 0.0 and heatmaps_batch.max() <= 1.0

save_image_heatmaps_batchs(img_batch, heatmaps_batch,
                           save_path="images/debug_sample0.png")

for i in range(10):
    img_batch_i, heatmaps_batch_i, _, _ = dataset[i]
    assert img_batch_i.shape == (3, 640, 640)
    assert heatmaps_batch_i.shape == (2, 160, 160)
print("10-sample loop passed")


loader = DataLoader(dataset, batch_size=4, num_workers=0)
img_batchs, heatmaps_batchs = next(iter(loader))
assert img_batchs.shape == (4, 3, 640, 640)
assert heatmaps_batchs.shape == (4, 2, 160, 160)
print("DataLoader batch test passed")

print('debug dataset end')
