from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kiloc.three_classifier.metrics import compute_classification_metrics


def train_classifier_one_epoch(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    dataloader: DataLoader,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    num_examples = 0

    for images, targets, _metadata in tqdm(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        num_examples += batch_size

    return {
        "train_loss": total_loss / max(num_examples, 1),
        "num_examples": num_examples,
    }


def evaluate_classifier(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device | str,
    dataloader: DataLoader,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    num_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for images, targets, metadata in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_labels = probs.argmax(axis=1)
            target_labels = targets.cpu().numpy()

            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            num_examples += batch_size

            y_true.extend(int(x) for x in target_labels.tolist())
            y_pred.extend(int(x) for x in pred_labels.tolist())

            for batch_index in range(batch_size):
                row = _build_prediction_row(
                    metadata=metadata,
                    batch_index=batch_index,
                    probs=probs[batch_index],
                    pred_label=int(pred_labels[batch_index]),
                    true_label=int(target_labels[batch_index]),
                )
                prediction_rows.append(row)

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=3)
    metrics["loss"] = total_loss / max(num_examples, 1)
    return metrics, prediction_rows


def _build_prediction_row(
    *,
    metadata: dict[str, object],
    batch_index: int,
    probs: np.ndarray,
    pred_label: int,
    true_label: int,
) -> dict[str, object]:
    prob_0 = float(probs[0])
    prob_1 = float(probs[1])
    prob_2 = float(probs[2])

    clipped = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    entropy = float(-(clipped * np.log(clipped)).sum())
    sorted_probs = np.sort(clipped)
    margin = float(sorted_probs[-1] - sorted_probs[-2])

    row = {
        "dataset_index": int(_get_meta_value(metadata, "dataset_index", batch_index)),
        "sample_id": str(_get_meta_value(metadata, "sample_id", batch_index)),
        "split": str(_get_meta_value(metadata, "split", batch_index)),
        "image_path": str(_get_meta_value(metadata, "image_path", batch_index)),
        "image_id": str(_get_meta_value(metadata, "image_id", batch_index)),
        "x": float(_get_meta_value(metadata, "x", batch_index)),
        "y": float(_get_meta_value(metadata, "y", batch_index)),
        "true_label": int(true_label),
        "pred_label": int(pred_label),
        "prob_class_0": prob_0,
        "prob_class_1": prob_1,
        "prob_class_2": prob_2,
        "entropy": entropy,
        "margin": margin,
        "is_correct": int(pred_label == true_label),
        "source": str(_get_meta_value(metadata, "source", batch_index)),
        "point_id": str(_get_meta_value(metadata, "point_id", batch_index)),
        "pred_id": str(_get_meta_value(metadata, "pred_id", batch_index)),
        "source_pred_class": str(_get_meta_value(metadata, "source_pred_class", batch_index)),
        "source_score": float(_get_meta_value(metadata, "source_score", batch_index)),
        "mining_tag": str(_get_meta_value(metadata, "mining_tag", batch_index)),
        "crop_size": int(_get_meta_value(metadata, "crop_size", batch_index)),
    }
    return row


def _get_meta_value(metadata: dict[str, object], key: str, batch_index: int) -> object:
    value = metadata[key]
    if torch.is_tensor(value):
        item = value[batch_index]
        return item.item() if item.ndim == 0 else item.tolist()
    if isinstance(value, np.ndarray):
        item = value[batch_index]
        return item.item() if np.ndim(item) == 0 else item.tolist()
    if isinstance(value, (list, tuple)):
        return value[batch_index]
    return value
