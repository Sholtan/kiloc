from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from kiloc.training.ema import ModelEMA


def _unpack_classifier_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Supports either:
      - tuple/list: (images, labels, ...)
      - dict: {"image": ..., "label": ...}
    """
    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch["label"]
        return images, labels

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                "Classifier batch must contain at least (images, labels)."
            )
        images = batch[0]
        labels = batch[1]
        return images, labels

    raise TypeError(
        f"Unsupported batch type: {type(batch)}. "
        "Expected dict or tuple/list."
    )


def _prepare_binary_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Converts labels to float tensor with the same shape as logits.
    Expected labels are 0/1.
    """
    labels = labels.float()

    # Common cases:
    # logits: [B], labels: [B]
    # logits: [B], labels: [B, 1]
    # logits: [B, 1], labels: [B]
    if labels.ndim > logits.ndim:
        labels = labels.squeeze(-1)
    elif logits.ndim > labels.ndim:
        labels = labels.unsqueeze(-1)

    if labels.shape != logits.shape:
        raise ValueError(
            f"Shape mismatch between logits {tuple(logits.shape)} "
            f"and labels {tuple(labels.shape)}."
        )

    return labels


def _compute_binary_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, float]:
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (
        2.0 * precision_pos * recall_pos / (precision_pos + recall_pos)
        if (precision_pos + recall_pos) > 0
        else 0.0
    )

    # Treat negative as its own class
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = (
        2.0 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0
        else 0.0
    )

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = 0.5 * (recall_pos + recall_neg)
    f1_macro = 0.5 * (f1_pos + f1_neg)
    precision_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_micro = (
        2.0 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "f1_macro": f1_macro,
        "precision": precision_micro,
        "recall": recall_micro,
        "f1": f1_micro,

        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def train_one_epoch(
    model: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    device: torch.device | str,
    trainloader: DataLoader,
    ema: ModelEMA | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = True,
    max_grad_norm: float | None = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    amp_enabled = use_amp and torch.cuda.is_available()
    amp_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext

    for batch in tqdm(trainloader):
        images, labels = _unpack_classifier_batch(batch)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx():
            logits = model(images)
            labels = _prepare_binary_targets(logits, labels)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()

    total_loss /= len(trainloader)
    return total_loss


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str,
    val_loader: DataLoader,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0

    tp = fp = fn = tn = 0

    amp_enabled = use_amp and torch.cuda.is_available()
    amp_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext

    for batch in tqdm(val_loader):
        images, labels = _unpack_classifier_batch(batch)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp_ctx():
            logits = model(images)
            labels_for_loss = _prepare_binary_targets(logits, labels)
            loss = criterion(logits, labels_for_loss)

        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        # Convert shapes to [B]
        if probs.ndim > 1:
            probs = probs.squeeze(-1)
        if labels.ndim > 1:
            labels = labels.squeeze(-1)

        preds = (probs >= threshold).long()
        labels = labels.long()

        tp += int(((preds == 1) & (labels == 1)).sum().item())
        fp += int(((preds == 1) & (labels == 0)).sum().item())
        fn += int(((preds == 0) & (labels == 1)).sum().item())
        tn += int(((preds == 0) & (labels == 0)).sum().item())

    total_loss /= len(val_loader)
    metrics = _compute_binary_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)

    return total_loss, metrics