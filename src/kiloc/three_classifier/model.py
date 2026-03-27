from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ThreeClassCropClassifier(nn.Module):
    def __init__(
        self,
        *,
        pretrained: bool = True,
        embedding_dim: int = 128,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.embedding = nn.Linear(in_features, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        features = self.backbone(x)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits
