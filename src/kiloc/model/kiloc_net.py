import torch
import torch.nn as nn
import torch.nn.functional as F


from kiloc.model.backbone import build_resnet34_backbone
from kiloc.model.fpn import FPN
from kiloc.model.head import HeatmapHead


class KiLocNet(nn.Module):
    """
    density model.

    Produces two localisation heatmaps (pos/neg)
    """

    def __init__(self, pretrained) -> None:
        super().__init__()
        self.fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)
        self.backbone = build_resnet34_backbone(pretrained=pretrained)

        # Heads
        self.pos_head = HeatmapHead(in_channels=256)
        self.neg_head = HeatmapHead(in_channels=256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone + FPN
        features = self.backbone(x)
        c2, c3, c4, c5 = features["c2"], features["c3"], features["c4"], features["c5"]
        p2, _, _, _ = self.fpn([c2, c3, c4, c5])  # use highest resolution

        # Predict localisation heatmaps
        loc_pos = self.pos_head(p2)
        loc_neg = self.neg_head(p2)

        # Stack channels
        loc_hm = torch.cat([loc_pos, loc_neg], dim=1)
        return loc_hm
