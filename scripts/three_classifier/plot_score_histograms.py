from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from kiloc.three_classifier.visualization import load_prediction_rows


CLASS_ID_TO_SHORT = {
    0: "ann_pos",
    1: "ann_neg",
    2: "mined",
}
SHORT_TO_CLASS_ID = {name: class_id for class_id, name in CLASS_ID_TO_SHORT.items()}

CLASS_ID_TO_SCORE_KEY = {
    0: "prob_class_0",
    1: "prob_class_1",
    2: "prob_class_2",
}

TRUE_LABEL_COLORS = {
    0: "#3366cc",
    1: "#2ca02c",
    2: "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot score histograms for ann_pos / ann_neg / mined probabilities "
            "from a saved 3-class classifier prediction CSV."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument(
        "--score-classes",
        nargs="+",
        choices=["ann_pos", "ann_neg", "mined"],
        default=["ann_pos", "ann_neg", "mined"],
        help="Which score columns to plot.",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-y", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_csv = args.predictions_csv or (args.run_dir / f"predictions_{args.split}.csv")
    rows = load_prediction_rows(predictions_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {predictions_csv}")

    score_class_ids = [SHORT_TO_CLASS_ID[name] for name in args.score_classes]
    fig, axes = plt.subplots(1, len(score_class_ids), figsize=(5.4 * len(score_class_ids), 4.8), constrained_layout=True)
    if len(score_class_ids) == 1:
        axes = [axes]
    bins = np.linspace(0.0, 1.0, int(args.bins) + 1)

    for class_id, ax in zip(score_class_ids, axes):
        score_key = CLASS_ID_TO_SCORE_KEY[class_id]
        for true_label in range(3):
            scores = np.asarray(
                [
                    float(row[score_key])
                    for row in rows
                    if int(row["true_label"]) == true_label
                ],
                dtype=np.float64,
            )
            ax.hist(
                scores,
                bins=bins,
                histtype="step",
                linewidth=1.8,
                color=TRUE_LABEL_COLORS[true_label],
                label=f"true {CLASS_ID_TO_SHORT[true_label]} (n={len(scores)})",
            )

        ax.set_title(f"score for {CLASS_ID_TO_SHORT[class_id]}")
        ax.set_xlabel("probability")
        ax.set_ylabel("count")
        ax.set_xlim(0.0, 1.0)
        if args.log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0].legend(loc="upper center", fontsize=9)
    fig.suptitle(
        f"{args.run_dir.name} | split={args.split} | score histograms by true label",
        fontsize=12,
    )

    out_path = args.out or (args.run_dir / "figures" / f"{args.split}_score_histograms.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out_path.as_posix())


if __name__ == "__main__":
    main()
