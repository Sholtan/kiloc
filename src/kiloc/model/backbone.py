import torch
import torch.nn
import torch.nn.functional as F

import torchvision.models as models

from torchvision.models.feature_extraction import create_feature_extractor

def build_resnet34_backbone(pretrained: bool = True) -> torch.nn.Module:
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet34(weights=weights)
    return create_feature_extractor(
        backbone,
        return_nodes={
            "layer1": "c2",  # stride 4,  channels 64
            "layer2": "c3",  # stride 8,  channels 128
            "layer3": "c4",  # stride 16, channels 256
            "layer4": "c5",  # stride 32, channels 512
        }
    )