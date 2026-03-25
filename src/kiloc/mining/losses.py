from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LocalHardNegativeBCELoss:
    eps: float = 1e-6

    def __call__(
        self,
        pred_logits: torch.Tensor,
        hardneg_weight_map: torch.Tensor,
    ) -> torch.Tensor:
        if pred_logits.shape != hardneg_weight_map.shape:
            raise ValueError(
                f"Shape mismatch: pred_logits={tuple(pred_logits.shape)} "
                f"hardneg_weight_map={tuple(hardneg_weight_map.shape)}"
            )

        weight_sum = hardneg_weight_map.sum()
        if float(weight_sum.detach().item()) <= self.eps:
            return pred_logits.new_zeros(())

        per_pixel_loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            torch.zeros_like(pred_logits),
            reduction="none",
        )
        return (per_pixel_loss * hardneg_weight_map).sum() / weight_sum.clamp_min(self.eps)


@dataclass
class LocalHardNegativeSigmoidMSELoss:
    eps: float = 1e-6

    def __call__(
        self,
        pred_logits: torch.Tensor,
        hardneg_weight_map: torch.Tensor,
    ) -> torch.Tensor:
        if pred_logits.shape != hardneg_weight_map.shape:
            raise ValueError(
                f"Shape mismatch: pred_logits={tuple(pred_logits.shape)} "
                f"hardneg_weight_map={tuple(hardneg_weight_map.shape)}"
            )

        weight_sum = hardneg_weight_map.sum()
        if float(weight_sum.detach().item()) <= self.eps:
            return pred_logits.new_zeros(())

        pred_prob = torch.sigmoid(pred_logits)
        per_pixel_loss = pred_prob.pow(2)
        return (per_pixel_loss * hardneg_weight_map).sum() / weight_sum.clamp_min(self.eps)


@dataclass
class LocalHardNegativeGeneralizedCrossEntropyLoss:
    q: float = 0.7
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not (0.0 < float(self.q) <= 1.0):
            raise ValueError(f"GCE q must be in (0, 1], got {self.q}")

    def __call__(
        self,
        pred_logits: torch.Tensor,
        hardneg_weight_map: torch.Tensor,
    ) -> torch.Tensor:
        if pred_logits.shape != hardneg_weight_map.shape:
            raise ValueError(
                f"Shape mismatch: pred_logits={tuple(pred_logits.shape)} "
                f"hardneg_weight_map={tuple(hardneg_weight_map.shape)}"
            )

        weight_sum = hardneg_weight_map.sum()
        if float(weight_sum.detach().item()) <= self.eps:
            return pred_logits.new_zeros(())

        pred_prob = torch.sigmoid(pred_logits)
        p_t = (1.0 - pred_prob).clamp_min(self.eps)
        per_pixel_loss = (1.0 - p_t.pow(float(self.q))) / float(self.q)
        return (per_pixel_loss * hardneg_weight_map).sum() / weight_sum.clamp_min(self.eps)


def build_hardneg_loss(name: str, *, gce_q: float = 0.7):
    if name == "local_bce_logits":
        return LocalHardNegativeBCELoss()
    if name == "local_sigmoid_mse":
        return LocalHardNegativeSigmoidMSELoss()
    if name == "local_gce":
        return LocalHardNegativeGeneralizedCrossEntropyLoss(q=float(gce_q))
    raise ValueError(
        "hardneg_loss_name must be one of "
        "{'local_bce_logits', 'local_sigmoid_mse', 'local_gce'}"
    )
