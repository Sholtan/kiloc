from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from kiloc.three_classifier.datasets import CLASS_ID_TO_NAME, ThreeClassCropDataset


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    *,
    num_classes: int = 3,
) -> dict[str, object]:
    true_arr = np.asarray(y_true, dtype=np.int64)
    pred_arr = np.asarray(y_pred, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_label, pred_label in zip(true_arr, pred_arr):
        confusion[int(true_label), int(pred_label)] += 1

    metrics: dict[str, object] = {
        "class_names": {str(class_id): CLASS_ID_TO_NAME[class_id] for class_id in range(num_classes)},
        "confusion_matrix": confusion.tolist(),
        "num_examples": int(len(true_arr)),
        "accuracy": float((true_arr == pred_arr).mean()) if len(true_arr) > 0 else 0.0,
    }

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    present_precision: list[float] = []
    present_recall: list[float] = []
    present_f1: list[float] = []

    for class_id in range(num_classes):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[f"support_class_{class_id}"] = support
        metrics[f"tp_class_{class_id}"] = tp
        metrics[f"fp_class_{class_id}"] = fp
        metrics[f"fn_class_{class_id}"] = fn
        metrics[f"precision_class_{class_id}"] = precision
        metrics[f"recall_class_{class_id}"] = recall
        metrics[f"f1_class_{class_id}"] = f1

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        if support > 0:
            present_precision.append(precision)
            present_recall.append(recall)
            present_f1.append(f1)

    metrics["precision_macro"] = float(np.mean(precision_values))
    metrics["recall_macro"] = float(np.mean(recall_values))
    metrics["f1_macro"] = float(np.mean(f1_values))
    metrics["precision_macro_present"] = float(np.mean(present_precision)) if present_precision else 0.0
    metrics["recall_macro_present"] = float(np.mean(present_recall)) if present_recall else 0.0
    metrics["f1_macro_present"] = float(np.mean(present_f1)) if present_f1 else 0.0
    return metrics


def write_prediction_csv(rows: list[dict[str, object]], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No prediction rows to write for {out_csv}")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_prediction_rows(
    rows: list[dict[str, object]],
    *,
    ranking: str,
) -> list[dict[str, object]]:
    if ranking == "entropy":
        return sorted(rows, key=lambda row: float(row["entropy"]), reverse=True)
    if ranking == "margin":
        return sorted(rows, key=lambda row: float(row["margin"]))
    raise ValueError(f"ranking must be one of {{'entropy', 'margin'}}, got {ranking}")


def save_hardest_samples(
    *,
    rows: list[dict[str, object]],
    dataset: ThreeClassCropDataset,
    out_dir: str | Path,
    ranking: str,
    top_k: int,
    export_images: bool,
) -> dict[str, object]:
    ranked_rows = rank_prediction_rows(rows, ranking=ranking)[:top_k]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hardest_csv = out_dir / f"hardest_{ranking}.csv"
    write_prediction_csv(ranked_rows, hardest_csv)

    image_dir = out_dir / f"hardest_{ranking}_images"
    if export_images:
        image_dir.mkdir(parents=True, exist_ok=True)
        for rank_index, row in enumerate(ranked_rows, start=1):
            dataset_index = int(row["dataset_index"])
            crop_rgb = dataset.get_crop_uint8(dataset_index)
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            filename = (
                f"{rank_index:04d}_"
                f"true{int(row['true_label'])}_"
                f"pred{int(row['pred_label'])}_"
                f"{str(row['image_id'])}_"
                f"{str(row['sample_id'])}.png"
            )
            cv2.imwrite(str(image_dir / filename), crop_bgr)

    summary = {
        "ranking": ranking,
        "top_k": len(ranked_rows),
        "csv_path": hardest_csv.as_posix(),
        "image_dir": image_dir.as_posix() if export_images else "",
        "export_images": bool(export_images),
    }
    with (out_dir / f"hardest_{ranking}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary
