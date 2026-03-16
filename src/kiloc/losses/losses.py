"""
Loss functions for project: Ki-67 index assesment using CNN backbone + FPN > localization head.

"""

import torch
from kiloc.utils.debug import print_info
from dataclasses import dataclass



def sigmoid_weighted_mse_loss(
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        pos_pts_tuple: tuple,
        neg_pts_tuple: tuple,
        alpha_pos: float = 100.0,
        alpha_neg: float = 100.0,
        q: float = 2.,
) -> torch.Tensor:
    """
    Weighted MSE for localization logits.

    Parameters
    ----------
    pred_logits: torch.Tensor
    Raw logits from localization head, shape (B, 2, 160, 160)

    target : torch.Tensor
    The ground truth heatmaps, shape (B, 2, 160, 160)

    alpha_pos: float
    Weighting factor for positive peaks; by default 100
    alpha_neg: float
    Weighting factor for negative peaks; by default 100

    Returns
    ----------

    total_loss: torch.Tensor (scalar)
    The scalar loss averaged over the batch, channels and pixels.
    """

    pred = torch.sigmoid(pred_logits)

    alpha = torch.tensor(
        [alpha_pos, alpha_neg],
        dtype=target.dtype,
        device=target.device,
    ).view(1, 2, 1, 1)
    w = 1.0 + alpha * torch.pow(target, q)   # shape: (B, 2, H, W)
    loss = w * (pred - target).pow(2)        # shape: (B, 2, H, W)

    # normalize per image, per channel
    loss = loss.sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-6)  # (B, 2)

    total_loss = loss.mean()
    return total_loss


def sigmoid_focal_loss(
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        pos_pts_tuple: tuple,
        neg_pts_tuple: tuple,
        alpha: float = 2.,
        beta: float = 4.,
):
    """
    focal loss for localization logits

    Parameters
    ----------
    pred_logits: torch.Tensor
    Raw logits from localization head, shape (B, 2, 160, 160)

    target : torch.Tensor
    The ground truth heatmaps, shape (B, 2, 160, 160)



    Returns
    ----------

    """
    number_of_cells = sum(len(x) for x in pos_pts_tuple) + \
        sum(len(x) for x in neg_pts_tuple)
    number_of_cells = max(number_of_cells, 1)

    pred = torch.sigmoid(pred_logits)
    pred = pred.clamp(min=1e-6, max=1-2e-6)

    loss1 = -1. * (1 - pred)**alpha * torch.log(pred)

    loss2 = -1. * (1 - target)**beta * pred**alpha * torch.log(1 - pred)

    mask = target >= 0.99999

    loss = torch.zeros(loss1.shape, dtype=pred.dtype, device=pred.device)

    loss[mask] = loss1[mask]
    loss[~mask] = loss2[~mask]

    loss = loss.sum(dim=(1, 2, 3)) / number_of_cells
    return loss.mean()



@dataclass
class SigmoidWeightedMSE:
    """
    Generates weighted MSE loss with certain alpha_pos and alpha_neg coefficients
    """
    alpha_pos: float = 100.0,
    alpha_neg: float = 100.0,
    q: float = 2.,

    def __call__(self, 
            pred_logits: torch.Tensor,
            target: torch.Tensor,
            pos_pts_tuple: tuple,
            neg_pts_tuple: tuple,)-> torch.Tensor: 
        """
        """
        pred = torch.sigmoid(pred_logits)

        alpha = torch.tensor(
            [self.alpha_pos, self.alpha_neg],
            dtype=target.dtype,
            device=target.device,
        ).view(1, 2, 1, 1)
        w = 1.0 + alpha * torch.pow(target, self.q)   # shape: (B, 2, H, W)
        loss = w * (pred - target).pow(2)        # shape: (B, 2, H, W)

        # normalize per image, per channel
        loss = loss.sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-6)  # (B, 2)

        total_loss = loss.mean()
        return total_loss
        