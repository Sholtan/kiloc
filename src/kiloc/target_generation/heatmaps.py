import torch
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class LocHeatmap:
    """
    Generates localization heatmap by taking maximum of Gaussians


    """

    out_hw: tuple[int, int]
    in_hw: tuple[int, int]
    sigma: float = 3.0
    dtype: torch.dtype = torch.float32

    def __call__(self, points_xy: NDArray[np.float32]) -> torch.Tensor:
        H, W = self.out_hw
        heatmap = torch.zeros((1, H, W), dtype=self.dtype)

        if len(points_xy) == 0:
            return heatmap

        pts = torch.Tensor(points_xy).to(self.dtype)

        # Rescale from input image coordinates to heatmap coordinates
        inH, inW = self.in_hw
        sx = W / float(inW)
        sy = H / float(inH)
        pts[:, 0] *= sx
        pts[:, 1] *= sy

        # grid coordinates
        yy = torch.arange(H, dtype=self.dtype).view(H, 1)
        xx = torch.arange(W, dtype=self.dtype).view(1, W)

        # Compute gaussian for each point of the grid, take max of all gaussians of cell points
        for x, y in pts:
            g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) /
                          (2.0 * self.sigma ** 2))
            heatmap[0] = torch.maximum(heatmap[0], g)

        return heatmap
