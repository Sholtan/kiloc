from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

SUPPORTED_BACKBONES = {"resnet18", "resnet34"}
SUPPORTED_INPUT_SIZES = {48, 96, 128}


def _build_resnet(backbone_name: str, pretrained: bool) -> nn.Module:
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=weights)

    if backbone_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet34(weights=weights)

    raise ValueError(
        f"Unsupported backbone_name={backbone_name!r}. "
        f"Choose from {sorted(SUPPORTED_BACKBONES)}."
    )


class ResNetClassifier(nn.Module):
    """
    Crop classifier for BCData point-level positive/negative classification.

    Parameters
    ----------
    backbone_name:
        One of {"resnet18", "resnet34"}.
    input_size:
        One of {48, 96, 128}. Used for validation and optional stem adaptation.
    num_classes:
        Use 1 for binary classification with BCEWithLogitsLoss.
        Use >1 if you later switch to softmax classification.
    pretrained:
        Whether to load ImageNet pretrained weights.
    dropout:
        Dropout before the final linear layer.
    adapt_small_inputs:
        If True, reduce early downsampling for 48x48 and 96x96 inputs
        by setting conv1 stride to 1 and removing maxpool.
        For 128x128, keep the default torchvision stem.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        input_size: int = 128,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.0,
        adapt_small_inputs: bool = True,
    ) -> None:
        super().__init__()

        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone_name={backbone_name!r}. "
                f"Choose from {sorted(SUPPORTED_BACKBONES)}."
            )

        if input_size not in SUPPORTED_INPUT_SIZES:
            raise ValueError(
                f"Unsupported input_size={input_size!r}. "
                f"Choose from {sorted(SUPPORTED_INPUT_SIZES)}."
            )

        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")

        self.backbone_name = backbone_name
        self.input_size = input_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = float(dropout)
        self.adapt_small_inputs = bool(adapt_small_inputs)

        backbone = _build_resnet(backbone_name=backbone_name, pretrained=pretrained)

        # Keep pretrained weights, but reduce early downsampling for smaller inputs.
        if self.adapt_small_inputs and input_size in {48, 96}:
            backbone.conv1.stride = (1, 1)
            backbone.maxpool = nn.Identity()

        in_features = backbone.fc.in_features

        if self.dropout > 0.0:
            backbone.fc = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)

        # Binary case: return shape [B] for BCEWithLogitsLoss
        if self.num_classes == 1:
            return logits.squeeze(1)

        # Multi-class case: return shape [B, C]
        return logits


def build_classifier(
    backbone_name: str,
    input_size: int,
    pretrained: bool = True,
    num_classes: int = 1,
    dropout: float = 0.0,
    adapt_small_inputs: bool = True,
) -> ResNetClassifier:
    return ResNetClassifier(
        backbone_name=backbone_name,
        input_size=input_size,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        adapt_small_inputs=adapt_small_inputs,
    )