from torch.utils.data import DataLoader
from kiloc import BCDataDataset

from pathlib import Path

import torch

from kiloc.utils.config import get_paths

from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.visualization.plots import save_image_heatmaps


print("start debug dataset")
# root_dir = Path(
#    "/home/yeldos/Istanbul/obsidian_vault/work_AITU/IHC/DATASETS/BCData")

root_dir, _ = get_paths("hpvictus")

heatmap_generator = LocHeatmap(out_hw=(160, 160), in_hw=(
    640, 640), sigma=3.0, dtype=torch.float32)


dataset = BCDataDataset(root=root_dir, split='train', target_transform=heatmap_generator,
                        image_transform=None, joint_transform=None)

print(f"len(dataset): {len(dataset)}")


img, heatmap = dataset[0]
assert img.shape == (3, 640, 640), img.shape
assert img.dtype == torch.float32, img.dtype
assert heatmap.shape == (2, 160, 160), heatmap.shape
assert heatmap.dtype == torch.float32, heatmap.dtype
assert img.min() >= 0.0 and img.max() <= 1.0
assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

save_image_heatmaps(img, heatmap, save_path="images/debug_sample0.png")

for i in range(10):
    img_i, heatmap_i = dataset[i]
    assert img_i.shape == (3, 640, 640)
    assert heatmap_i.shape == (2, 160, 160)
print("10-sample loop passed")


loader = DataLoader(dataset, batch_size=4, num_workers=0)
imgs, heatmaps = next(iter(loader))
assert imgs.shape == (4, 3, 640, 640)
assert heatmaps.shape == (4, 2, 160, 160)
print("DataLoader batch test passed")

print('debug dataset end')
