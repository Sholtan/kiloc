from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved 3-class confusion matrix.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _short_label(name: str) -> str:
    return {
        "annotated_positive": "ann_pos",
        "annotated_negative": "ann_neg",
        "mined_rexclude30": "mined",
    }.get(name, name)


def _draw_matrix(
    *,
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    labels: list[str],
    fmt: str,
    cmap: str,
) -> None:
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)

    max_value = float(matrix.max()) if matrix.size > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if max_value > 0 and value > 0.6 * max_value else "black"
            ax.text(
                col,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    results_json = args.run_dir / f"{args.split}_results.json"
    with results_json.open() as handle:
        data = json.load(handle)

    confusion = np.asarray(data["confusion_matrix"], dtype=np.float64)
    class_names = [
        _short_label(data["class_names"][str(class_id)])
        for class_id in range(confusion.shape[0])
    ]

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion),
        where=row_sums > 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _draw_matrix(
        ax=axes[0],
        matrix=confusion,
        title=f"{args.split.title()} confusion (counts)",
        labels=class_names,
        fmt=".0f",
        cmap="Blues",
    )
    _draw_matrix(
        ax=axes[1],
        matrix=normalized,
        title=f"{args.split.title()} confusion (row-normalized)",
        labels=class_names,
        fmt=".2f",
        cmap="Greens",
    )

    summary = (
        f"macro F1={data['f1_macro']:.3f} | "
        f"class F1s=({data['f1_class_0']:.3f}, {data['f1_class_1']:.3f}, {data['f1_class_2']:.3f})"
    )
    fig.suptitle(summary, fontsize=12)

    out_path = args.out or (args.run_dir / "figures" / f"{args.split}_confusion_matrix.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
