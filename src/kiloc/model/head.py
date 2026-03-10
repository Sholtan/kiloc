import torch
import torch.nn as nn


class HeatmapHead(nn.Module):
    """
    A simple convolutional head that predicts a single‑channel heatmap from a
    feature map. Consists of a stack of 3×3 convs followed by a 1×1 conv.
    """

    def __init__(self, in_channels: int, num_convs: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.out(x)
        return x
