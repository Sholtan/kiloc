import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
from kiloc.model.kiloc_net import KiLocNet


def train_one_epoch(
        model: KiLocNet,
        criterion: Callable,
        optimizer: Optimizer,
        device: torch.device,
        trainloader: DataLoader
):
    model.train()
    total_loss = 0.0
    for img_batch, heatmaps_batch, pos_pts_tuple, neg_pts_tuple in tqdm(trainloader):
        optimizer.zero_grad()

        img_batch = img_batch.to(device, non_blocking=True)
        heatmaps_batch = heatmaps_batch.to(device, non_blocking=True)

        pred_logits = model(img_batch)

        loss = criterion(pred_logits, heatmaps_batch,
                         pos_pts_tuple, neg_pts_tuple)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    total_loss /= len(trainloader)

    return total_loss
