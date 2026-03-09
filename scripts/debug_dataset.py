from kiloc import BCDataDataset

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import torch

print("start debug dataset")
root_dir = Path(
    "/home/yeldos/Istanbul/obsidian_vault/work_AITU/IHC/DATASETS/BCData")


def dummy_transform(pts: NDArray[np.int64]):
    return np.zeros((160, 160))


dataset = BCDataDataset(root=root_dir, split='train', target_transform=dummy_transform,
                        image_transform=None, joint_transform=None)

print(f"len(dataset): {len(dataset)}")

img_tensor, heatmap_tensor = dataset[0]

assert isinstance(img_tensor, torch.Tensor)
assert img_tensor.shape == (3, 640, 640)


print('\n\ndone')
