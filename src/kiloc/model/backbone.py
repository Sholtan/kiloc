import torch
import torch.nn
import torch.nn.functional as F

import torchvision.models as models

from torchvision.models.feature_extraction import create_feature_extractor

BACKBONE_CHANNELS = {
    'resnet34': [64, 128, 256, 512],
    'resnet50': [256, 512, 1024, 2048],
}

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

def build_resnet50_backbone(pretrained: bool = True) -> torch.nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet50(weights=weights)

    return create_feature_extractor(backbone, return_nodes={
        "layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5",
    })

def build_backbone(name: str, pretrained: bool = True) -> torch.nn.Module:
    if name == 'resnet34': return build_resnet34_backbone(pretrained)
    if name == 'resnet50': return build_resnet50_backbone(pretrained)
    raise ValueError(f"Unknown backbone: {name}")