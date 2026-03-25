import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.model.kiloc_net import KiLocNet
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.utils.config import get_paths


# ImageNet normalization stats
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Support either raw state_dict or wrapped checkpoint dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)


def tensor_to_rgb_image(img_tensor: torch.Tensor, input_normalization) -> np.ndarray:
    """
    Convert CHW tensor to HWC float image in [0, 1] for plotting.
    """
    x = img_tensor.detach().cpu().float().clone()

    if input_normalization == "imagenet":
        x = x * IMAGENET_STD + IMAGENET_MEAN
        x = x.clamp(0.0, 1.0)
    elif input_normalization in [None, "none", "identity", ""]:
        x = x.clamp(0.0, 1.0)
    else:
        # Fallback if normalization mode is unknown
        x_min = x.min()
        x_max = x.max()
        if (x_max - x_min) > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = torch.zeros_like(x)

    return x.permute(1, 2, 0).numpy()


def filter_not_annotated_candidates(pred_pts: np.ndarray,
                                    gt_pos: np.ndarray,
                                    gt_neg: np.ndarray,
                                    exclude_radius: float) -> np.ndarray:
    """
    Keep only predictions farther than exclude_radius from every GT point
    of either class.
    """
    pred_pts = np.asarray(pred_pts, dtype=np.float32).reshape(-1, 2)
    gt_pos = np.asarray(gt_pos, dtype=np.float32).reshape(-1, 2)
    gt_neg = np.asarray(gt_neg, dtype=np.float32).reshape(-1, 2)

    if len(pred_pts) == 0:
        return pred_pts.copy()

    gt_all = []
    if len(gt_pos) > 0:
        gt_all.append(gt_pos)
    if len(gt_neg) > 0:
        gt_all.append(gt_neg)

    if len(gt_all) == 0:
        return pred_pts.copy()

    gt_all = np.concatenate(gt_all, axis=0)  # [M, 2]

    # Pairwise distances from predictions to all GT points
    d2 = ((pred_pts[:, None, :] - gt_all[None, :, :]) ** 2).sum(axis=2)  # [N, M]
    min_d = np.sqrt(d2.min(axis=1))
    keep_mask = min_d > exclude_radius

    return pred_pts[keep_mask]


def scatter(ax, pts: np.ndarray, color: str, label: str, s: int = 14) -> None:
    if len(pts) > 0:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            s=s,
            label=label,
            linewidths=0,
        )


def resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        ckpt_path = run_dir / checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    ckpt_paths = sorted(run_dir.glob("*.pth"))
    if len(ckpt_paths) != 1:
        raise RuntimeError(f"Expected exactly 1 .pth in {run_dir}, found: {ckpt_paths}")
    return ckpt_paths[0]


def main(run_dir: Path,
         n_images: int | None,
         split: str,
         checkpoint: str | None,
         threshold: float,
         exclude_radius: float,
         out_name: str | None) -> None:
    with open(run_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = resolve_checkpoint(run_dir, checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = KiLocNet(pretrained=False, backbone_name=cfg["backbone"])
    load_model_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()

    data_root, _ = get_paths(device="h200")

    heatmap_gen = LocHeatmap(
        out_hw=cfg["out_hw"],
        in_hw=cfg["in_hw"],
        sigma=cfg["sigma"],
        dtype=torch.float32,
    )

    dataset = BCDataDataset(
        root=data_root,
        split=split,
        target_transform=heatmap_gen,
        input_normalization=cfg["input_normalization"],
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    suffix = out_name if out_name is not None else f"not_annotated_vis_{split}"
    out_dir = run_dir / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel_size = cfg["kernel_size"]
    merge_radius = cfg["merge_radius"]

    saved = 0
    seen = 0

    with torch.no_grad():
        for idx, (img_batch, _, pos_pts_tuple, neg_pts_tuple) in enumerate(loader):
            if n_images is not None and seen >= n_images:
                break
            seen += 1

            img_batch = img_batch.to(device)
            logits = model(img_batch)
            pred_heatmaps = torch.sigmoid(logits)

            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=pred_heatmaps,
                kernel_size=kernel_size,
                threshold=threshold,
                merge_radius=merge_radius,
            )

            pred_pos = np.asarray(out_pos[0], dtype=np.float32).reshape(-1, 2)
            pred_neg = np.asarray(out_neg[0], dtype=np.float32).reshape(-1, 2)

            gt_pos = np.asarray(pos_pts_tuple[0], dtype=np.float32).reshape(-1, 2)
            gt_neg = np.asarray(neg_pts_tuple[0], dtype=np.float32).reshape(-1, 2)

            not_annotated_pos = filter_not_annotated_candidates(
                pred_pts=pred_pos,
                gt_pos=gt_pos,
                gt_neg=gt_neg,
                exclude_radius=exclude_radius,
            )

            not_annotated_neg = filter_not_annotated_candidates(
                pred_pts=pred_neg,
                gt_pos=gt_pos,
                gt_neg=gt_neg,
                exclude_radius=exclude_radius,
            )

            n_not_annotated = len(not_annotated_pos) + len(not_annotated_neg)
            if n_not_annotated == 0:
                continue

            img_np = tensor_to_rgb_image(
                img_tensor=img_batch[0],
                input_normalization=cfg.get("input_normalization", None),
            )

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))

            # Left: mined "not annotated" candidates
            ax_left.imshow(img_np)
            scatter(ax_left, not_annotated_pos, color="red", label="Not annotated (pred pos)")
            scatter(ax_left, not_annotated_neg, color="orange", label="Not annotated (pred neg)")
            ax_left.set_title(
                f"Not annotated candidates | total={n_not_annotated} "
                f"(pos={len(not_annotated_pos)}, neg={len(not_annotated_neg)})"
            )
            if n_not_annotated > 0:
                ax_left.legend(loc="upper right")
            ax_left.axis("off")

            # Right: GT annotations
            ax_right.imshow(img_np)
            scatter(ax_right, gt_pos, color="lime", label="GT pos")
            scatter(ax_right, gt_neg, color="deepskyblue", label="GT neg")
            ax_right.set_title(
                f"GT annotations | total={len(gt_pos) + len(gt_neg)} "
                f"(pos={len(gt_pos)}, neg={len(gt_neg)})"
            )
            if (len(gt_pos) + len(gt_neg)) > 0:
                ax_right.legend(loc="upper right")
            ax_right.axis("off")

            fig.suptitle(
                f"Image {idx:03d} | threshold={threshold:.3f} | exclude_radius={exclude_radius:.1f}",
                fontsize=12
            )

            out_path = out_dir / f"image_{idx:03d}_not_annotated_{n_not_annotated:03d}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved += 1
            print(f"Saved {out_path.name}")

    print(f"Done. Saved {saved} images to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--exclude_radius",
        type=float,
        default=30.0,
        help="Exclude any prediction within this radius of any GT point."
    )
    parser.add_argument(
        "--out_name",
        default=None,
        help="Optional custom output directory name inside run_dir."
    )

    args = parser.parse_args()

    main(
        run_dir=args.run_dir,
        n_images=args.n_images,
        split=args.split,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        exclude_radius=args.exclude_radius,
        out_name=args.out_name,
    )