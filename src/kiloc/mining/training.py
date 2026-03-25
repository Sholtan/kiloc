from __future__ import annotations

from collections.abc import Callable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from kiloc.model.kiloc_net import KiLocNet
from kiloc.training.ema import ModelEMA


def train_one_epoch_hardneg(
    *,
    model: KiLocNet,
    base_criterion: Callable,
    hardneg_criterion: Callable,
    lambda_hardneg: float,
    optimizer: Optimizer,
    device: torch.device | str,
    trainloader: DataLoader,
    ema: ModelEMA | None = None,
) -> dict[str, float | int]:
    model.train()

    total_loss_sum = 0.0
    base_loss_sum = 0.0
    hardneg_loss_raw_sum = 0.0
    hardneg_loss_weighted_sum = 0.0
    hardneg_weight_sum_total = 0.0
    active_hardneg_images = 0
    num_images = 0

    for img_batch, heatmaps_batch, pos_pts_tuple, neg_pts_tuple, hardneg_weight_batch in tqdm(trainloader):
        optimizer.zero_grad()

        img_batch = img_batch.to(device, non_blocking=True)
        heatmaps_batch = heatmaps_batch.to(device, non_blocking=True)
        hardneg_weight_batch = hardneg_weight_batch.to(device, non_blocking=True)

        pred_logits = model(img_batch)

        base_loss = base_criterion(pred_logits, heatmaps_batch, pos_pts_tuple, neg_pts_tuple)
        hardneg_loss = hardneg_criterion(pred_logits, hardneg_weight_batch)
        total_loss = base_loss + float(lambda_hardneg) * hardneg_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss_sum += float(total_loss.item())
        base_loss_sum += float(base_loss.item())
        hardneg_loss_raw_sum += float(hardneg_loss.item())
        hardneg_loss_weighted_sum += float(lambda_hardneg) * float(hardneg_loss.item())
        hardneg_weight_sum_total += float(hardneg_weight_batch.sum().item())
        active_hardneg_images += int((hardneg_weight_batch.sum(dim=(1, 2, 3)) > 0).sum().item())
        num_images += int(img_batch.shape[0])

    num_batches = len(trainloader)
    train_base_loss = base_loss_sum / num_batches
    train_hardneg_loss_raw = hardneg_loss_raw_sum / num_batches
    train_hardneg_loss = hardneg_loss_weighted_sum / num_batches
    return {
        "train_loss": total_loss_sum / num_batches,
        "train_base_loss": train_base_loss,
        "train_hardneg_loss": train_hardneg_loss,
        "train_hardneg_loss_raw": train_hardneg_loss_raw,
        "train_hardneg_to_base_ratio": (
            train_hardneg_loss / train_base_loss if train_base_loss > 0 else 0.0
        ),
        "hardneg_weight_sum_total": hardneg_weight_sum_total,
        "hardneg_active_images": active_hardneg_images,
        "num_images": num_images,
    }
