from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from kiloc.datasets.bcdata import BCDataDataset, collate_fn


class BCDataHardNegativeDataset(Dataset):
    def __init__(
        self,
        base_dataset: BCDataDataset,
        *,
        hardneg_by_image: dict[str, list[dict]],
        hardneg_target_transform: Callable[[list[dict]], torch.Tensor],
    ) -> None:
        if base_dataset.joint_transform is not None:
            raise ValueError(
                "BCDataHardNegativeDataset currently requires base_dataset.joint_transform is None. "
                "Use augmentation=false for the first hard-negative fine-tuning stage."
            )

        self.base_dataset = base_dataset
        self.hardneg_by_image = hardneg_by_image
        self.hardneg_target_transform = hardneg_target_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img_tensor, heatmaps, pos_pts, neg_pts = self.base_dataset[idx]
        image_path, _, _ = self.base_dataset.samples[idx]
        image_id = image_path.stem

        hardneg_rows = self.hardneg_by_image.get(image_id, [])
        hardneg_weight_map = self.hardneg_target_transform(hardneg_rows)
        return img_tensor, heatmaps, pos_pts, neg_pts, hardneg_weight_map


def collate_fn_hardneg(batch):
    base_items = [(img, heatmaps, pos_pts, neg_pts) for img, heatmaps, pos_pts, neg_pts, _ in batch]
    img_batch, heatmaps_batch, pos_pts_tuple, neg_pts_tuple = collate_fn(base_items)
    hardneg_batch = torch.stack([hardneg_weight_map for _, _, _, _, hardneg_weight_map in batch], dim=0)
    return img_batch, heatmaps_batch, pos_pts_tuple, neg_pts_tuple, hardneg_batch
