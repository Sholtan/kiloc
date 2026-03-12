import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
import numpy as np
from tqdm import tqdm

from kiloc.model.kiloc_net import KiLocNet
from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.utils.debug import print_info
from kiloc.evaluation.metrics import compute_metrics


def train_one_epoch(
        model: KiLocNet,
        criterion: Callable,
        optimizer: Optimizer,
        device: torch.device | str,
        trainloader: DataLoader
) -> float:
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


def val_one_epoch(
    model: KiLocNet,
    criterion: Callable,
    device: torch.device | str,
    val_loader: DataLoader,
    kernel_size: int = 3,
    threshold: float = 0.5,
    merge_radius: float = 1.5,
    matching_radius: float = 6.0,
) -> tuple[float, float, float, float]:    # val_loss, precision, recall, f1
    model.eval()
    total_loss = 0.

    tp_pos = fp_pos = fn_pos = 0
    tp_neg = fp_neg = fn_neg = 0
    with torch.no_grad():

        for img_batch, heatmaps_batch, pos_pts_tuple, neg_pts_tuple in tqdm(val_loader):
            img_batch = img_batch.to(device, non_blocking=True)
            heatmaps_batch = heatmaps_batch.to(device, non_blocking=True)

            logits = model(img_batch)
            loss = criterion(logits, heatmaps_batch,
                             pos_pts_tuple, neg_pts_tuple)
            total_loss += loss.item()

            pred_heatmaps = torch.sigmoid(logits)

            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=pred_heatmaps, kernel_size=kernel_size, threshold=threshold, merge_radius=merge_radius)

            print_info(pos_pts_tuple, "pos_pts_tuple")
            print_info(out_pos, "out_pos")

            for pred, gt in zip(out_pos, pos_pts_tuple):
                tp, fp, fn = compute_metrics(pred, gt.astype(
                    np.float32), radius=matching_radius)
                tp_pos += tp
                fp_pos += fp
                fn_pos += fn

            for pred, gt in zip(out_neg, neg_pts_tuple):
                tp, fp, fn = compute_metrics(pred, gt.astype(
                    np.float32), radius=matching_radius)
                tp_neg += tp
                fp_neg += fp
                fn_neg += fn

    tp_both = tp_pos + tp_neg
    fp_both = fp_pos + fp_neg
    fn_both = fn_pos + fn_neg

    precision_both = tp_both / \
        (tp_both + fp_both) if (tp_both + fp_both) > 0 else 0.0
    recall_both = tp_both / \
        (tp_both + fn_both) if (tp_both + fn_both) > 0 else 0.0
    f1_both = 2 * precision_both * recall_both / \
        (precision_both + recall_both) if (precision_both + recall_both) > 0 else 0.0

    total_loss /= len(val_loader)
    return total_loss, precision_both, recall_both, f1_both
