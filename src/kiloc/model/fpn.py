import torch
import torch.nn as nn
import torch.nn.functional as F






class FPN(nn.Module):
    """
    Feature Pyramid Network (1×1 lateral connections + top‑down upsampling).

    Only the highest resolution (P2) is used downstream by the heads, but all
    pyramid levels are returned for potential future use.
    """
    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        # Lateral 1×1 projections from C2–C5 into a fixed number of channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        # 3×3 convolutions applied after merging top‑down features
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c2, c3, c4, c5 = features
        # top‑down pathway
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        # output convolutions
        p5 = self.output_convs[3](p5)
        p4 = self.output_convs[2](p4)
        p3 = self.output_convs[1](p3)
        p2 = self.output_convs[0](p2)
        return p2, p3, p4, p5