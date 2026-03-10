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

    image_transform: Callable
        Image augmentations that don't require modification of the target.

    joint_transform: Callable
        Augmnentations that require modification of both image and target. eg. rotations.
    """
    SUPPORTED_SPLITS = {"train", "validation", "test"}

    def __init__(
            self,
            root: Path,
            split: str,
            target_transform: Callable,
            image_transform: Callable | None = None,
            joint_transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.target_transform = target_transform
        self.image_transform = image_transform
        self.joint_transform = joint_transform

        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Unknown split name, choose one from: {self.SUPPORTED_SPLITS}")
        if not callable(target_transform):
            raise TypeError(
                "target_transform must be callable (carries out heatmap generation)")

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

        for img_path in image_paths:
            stem = img_path.stem   # the image name without .png extension

            pos_ann_path = self.pos_ann_dir / (stem + ".h5")
            neg_ann_path = self.neg_ann_dir / (stem + ".h5")

            if not pos_ann_path.exists():
                raise FileNotFoundError(f"Missing annotation: {pos_ann_path}")
            if not neg_ann_path.exists():
                raise FileNotFoundError(f"Missing annotation: {neg_ann_path}")

            samples.append((img_path, pos_ann_path, neg_ann_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_points(self, ann_path: Path) -> NDArray[np.int64]:
        with h5py.File(ann_path, 'r') as f:
            dset = f["coordinates"]
            assert isinstance(dset, h5py.Dataset)
            coords = dset[:]
        return coords

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads sample image and its positive and negative cells annotations. Converts annotations to heatmaps.

        Returns
        -------
        image: torch.Tensor (3, H, W) float32
        heatmap: (2, H', W') float32
        """
        img_path, pos_ann_path, neg_ann_path = self.samples[idx]

        img: NDArray[np.uint8] = cv2.imread(
            str(img_path))    # pixel value range: 0 - 255, BGR, (H, W, C)
        if img is None:
            raise RuntimeError(f"Failed to read image at {img_path}")

        img_scaled: NDArray[np.float32] = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

        img_tensor: torch.Tensor = torch.from_numpy(
            img_scaled).permute(2, 0, 1).contiguous()

        # Load point annotations as numpy array (shape (N, 2))
        pos_pts: NDArray[np.int64] = self._load_points(pos_ann_path)
        neg_pts: NDArray[np.int64] = self._load_points(neg_ann_path)

        # Apply joint transform (e.g. random rotation) before any image-only
        # transformation. This transform must return a new image tensor and
        # updated point coordinates. If no joint transform is provided,
        # image and points lists remain umchanged.
        if self.joint_transform is not None:
            img_tensor, pos_pts, neg_pts = self.joint_transform(
                img_tensor, pos_pts, neg_pts)

        # Apply image-only transform if provided.
        if self.image_transform is not None:
            img_tensor = self.image_transform(img_tensor)

        # Generate localization heatmaps (peak-normalized) for pos and neg annotations
        pos_heatmap = self.target_transform(pos_pts)
        neg_heatmap = self.target_transform(neg_pts)

        heatmap = torch.cat([pos_heatmap, neg_heatmap], dim=0)
        return img_tensor, heatmap
