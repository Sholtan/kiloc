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
from kiloc.training.ema import ModelEMA
from kiloc.evaluation.tta import tta_forward


def train_one_epoch(
        model: KiLocNet,
        criterion: Callable,
        optimizer: Optimizer,
        device: torch.device | str,
        trainloader: DataLoader,
        ema: ModelEMA | None = None,
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

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()

    total_loss /= len(trainloader)

    return total_loss


def val_one_epoch(
    model: KiLocNet,
    criterion: Callable,
    device: torch.device | str,
    val_loader: DataLoader,
    kernel_size: int = 3,
    threshold: float | tuple[float, float] = 0.5,
    merge_radius: float = 1.5,
    matching_radius: float = 6.0,
    tta: bool = False
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
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

            #pred_heatmaps = torch.sigmoid(logits)
            pred_heatmaps = tta_forward(model, img_batch) if tta else torch.sigmoid(logits)

            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=pred_heatmaps, kernel_size=kernel_size, threshold=threshold, merge_radius=merge_radius)

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



    # Compute metrics for positive
    precision_pos = tp_pos / \
        (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
    recall_pos = tp_pos / \
        (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0.0
    f1_pos = 2 * precision_pos * recall_pos / \
        (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0.0

    # Compute metrics for negative
    precision_neg = tp_neg / \
        (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0.0
    recall_neg = tp_neg / \
        (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else 0.0
    f1_neg = 2 * precision_neg * recall_neg / \
        (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0.0

    # Compute overall metrics
    tp_both = tp_pos + tp_neg
    fp_both = fp_pos + fp_neg
    fn_both = fn_pos + fn_neg

    # micro
    precision_both = tp_both / \
        (tp_both + fp_both) if (tp_both + fp_both) > 0 else 0.0
    recall_both = tp_both / \
        (tp_both + fn_both) if (tp_both + fn_both) > 0 else 0.0
    f1_both = 2 * precision_both * recall_both / \
        (precision_both + recall_both) if (precision_both + recall_both) > 0 else 0.0
    
    #macro
    precision_macro = 0.5 * (precision_pos + precision_neg)
    recall_macro = 0.5 * (recall_pos + recall_neg)
    f1_macro = 0.5 * (f1_pos + f1_neg)

    total_loss /= len(val_loader)
    return total_loss, precision_both, recall_both, f1_both, \
        precision_pos, recall_pos, f1_pos, \
        precision_neg, recall_neg, f1_neg, f1_macro
