"""
Dataset implimentation for BCData Ki-67 dataset.
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import h5py

import numpy as np
from numpy.typing import NDArray

from typing import Any
from collections.abc import Callable

# IMAGENET statistics, used for normalization when using backbones trained on IMAGENET
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

class AlbumentationsJointTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, pos_pts, neg_pts):
        n_pos = len(pos_pts)
        all_kps = pos_pts.tolist() + neg_pts.tolist()

        result = self.transform(image=img, keypoints=all_kps)
        aug_img = result['image']
        aug_kps = result['keypoints']

        aug_pos = np.array(aug_kps[:n_pos])   if n_pos > 0          else np.empty((0, 2), dtype=np.float32)
        aug_neg = np.array(aug_kps[n_pos:])   if len(aug_kps) > n_pos else np.empty((0, 2), dtype=np.float32)

        # filter out-of-bounds points (remove_invisible=False keeps them)
        h, w = aug_img.shape[:2]
        if len(aug_pos) > 0:
            mask = (aug_pos[:, 0] >= 0) & (aug_pos[:, 0] < w) & \
                   (aug_pos[:, 1] >= 0) & (aug_pos[:, 1] < h)
            aug_pos = aug_pos[mask]
        if len(aug_neg) > 0:
            mask = (aug_neg[:, 0] >= 0) & (aug_neg[:, 0] < w) & \
                   (aug_neg[:, 1] >= 0) & (aug_neg[:, 1] < h)
            aug_neg = aug_neg[mask]

        return aug_img, aug_pos, aug_neg


class BCDataDataset(Dataset):
    """
    Parses the images and annotations for the BCData dataset.
    Parameters
    ----------
    root:Path
        Root directory where dataset is located. Must have 'annotations', 'images' subdirectories.

    split: str
        One of sets. Determines which folder to read.

    target_transform: Callable
        Generation of heatmaps. accepts * and returns *

    joint_transform: Callable
        Augmnentations that require modification of both image and target. eg. rotations.
    """
    SUPPORTED_SPLITS = {"train", "validation", "test"}

    def __init__(
            self,
            root: Path,
            split: str,
            target_transform: Callable,
            joint_transform: Callable | None = None,
            input_normalization: str | None = None,
            image_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.input_normalization = input_normalization
        self.image_ids = None if image_ids is None else {str(image_id) for image_id in image_ids}

        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Unknown split name, choose one from: {self.SUPPORTED_SPLITS}")
        if not callable(target_transform):
            raise TypeError(
                "target_transform must be callable (carries out heatmap generation)")
        if self.image_ids is not None and len(self.image_ids) == 0:
            raise ValueError("image_ids must not be empty if provided")

        self.image_dir = self.root / "images" / self.split
        self.pos_ann_dir = self.root / "annotations" / self.split / "positive"
        self.neg_ann_dir = self.root / "annotations" / self.split / "negative"

        self.samples: list[tuple[Path, Path, Path]] = self._build_index()

    def _build_index(self) -> list[tuple[Path, Path, Path]]:
        """
        Returns a list of pairs (sample's image directory, sample's pos ann directory, sample's neg ann directory)
        """
        image_paths = sorted(self.image_dir.glob("*.png"))
        if not image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        samples: list[tuple[Path, Path, Path]] = []
        found_image_ids: set[str] = set()

        for img_path in image_paths:
            stem = img_path.stem   # the image name without .png extension
            if self.image_ids is not None and stem not in self.image_ids:
                continue

            pos_ann_path = self.pos_ann_dir / (stem + ".h5")
            neg_ann_path = self.neg_ann_dir / (stem + ".h5")

            if not pos_ann_path.exists():
                raise FileNotFoundError(f"Missing annotation: {pos_ann_path}")
            if not neg_ann_path.exists():
                raise FileNotFoundError(f"Missing annotation: {neg_ann_path}")

            samples.append((img_path, pos_ann_path, neg_ann_path))
            found_image_ids.add(stem)

        if self.image_ids is not None:
            missing = sorted(self.image_ids.difference(found_image_ids))
            if missing:
                raise FileNotFoundError(
                    f"Requested image_ids were not found in split={self.split}: {missing[:10]}"
                )
            if not samples:
                raise RuntimeError(f"No samples matched image_ids for split={self.split}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_points(self, ann_path: Path) -> NDArray[np.int64]:
        with h5py.File(ann_path, 'r') as f:
            dset = f["coordinates"]
            assert isinstance(dset, h5py.Dataset)
            coords = dset[:]
        return coords

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, NDArray[np.int64], NDArray[np.int64]]:
        """
        Loads sample image and its positive and negative cells annotations. Converts annotations to heatmaps.

        Returns
        -------
        image: torch.Tensor (3, H, W) float32
        heatmap: (2, H', W') float32
        pos_pts: NDArray (N, 2) np.int64, numpy array with pixel coordinates of positive cells
        neg_pts: NDArray (N, 2) np.int64, numpy array with pixel coordinates of negative cells
        """
        img_path, pos_ann_path, neg_ann_path = self.samples[idx]

        img: NDArray[np.uint8] = cv2.imread(
            str(img_path))    # pixel value range: 0 - 255, BGR, (H, W, C)
        if img is None:
            raise RuntimeError(f"Failed to read image at {img_path}")

        img_scaled: NDArray[np.float32] = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

        # Load point annotations as numpy array (shape (N, 2))
        pos_pts: NDArray[np.int64] = self._load_points(pos_ann_path)
        neg_pts: NDArray[np.int64] = self._load_points(neg_ann_path)


        # Apply joint transform (e.g. random rotation) before any image-only
        # transformation. This transform must return a new image tensor and
        # updated point coordinates. If no joint transform is provided,
        # image and points lists remain umchanged.
        if self.joint_transform is not None:
            img_scaled, pos_pts, neg_pts = self.joint_transform(img_scaled, pos_pts, neg_pts)

        img_tensor: torch.Tensor = (
            torch.from_numpy(img_scaled)
            .permute(2, 0, 1)
            .contiguous()
            .float()
        )
        if self.input_normalization == 'imagenet':
            img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
        




        # Generate localization heatmaps (peak-normalized) for pos and neg annotations
        pos_heatmap = self.target_transform(pos_pts)
        neg_heatmap = self.target_transform(neg_pts)

        heatmaps = torch.cat([pos_heatmap, neg_heatmap], dim=0)
        return img_tensor, heatmaps, pos_pts, neg_pts


def collate_fn(batch):
    """
    Custom collate function used by Dataloader

    """
    img_tuple, heatmaps_tuple, pos_pts_tuple, neg_pts_tuple = zip(*batch)

    img_batch = torch.stack(img_tuple, dim=0)  # (B, 3, H, W)
    heatmaps_batch = torch.stack(heatmaps_tuple, dim=0)

    return img_batch, heatmaps_batch, pos_pts_tuple, neg_pts_tuple
