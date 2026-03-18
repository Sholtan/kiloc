import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from kiloc.utils.config import get_paths
from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.evaluation.metrics import match_points
from torch.utils.data import DataLoader

# IMAGENET statistics, used for normalization when using backbones trained on IMAGENET
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def main(run_dir, n_images, split, checkpoint):
    with open(run_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)

    if checkpoint is not None:
        print("checkpoint is not none")
        ckpt_paths = [run_dir / checkpoint]
    else:
        print("checkpoint is none")
        ckpt_paths = list(run_dir.glob('*.pth'))
        assert len(ckpt_paths) == 1, f"Expected 1 checkpoint, found {ckpt_paths}"
    
    ckpt_path = ckpt_paths[0]

    model = KiLocNet(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    root_dir, _ = get_paths(device='h200')

    heatmap_gen = LocHeatmap(
        out_hw=cfg['out_hw'], in_hw=cfg['in_hw'],
        sigma=cfg['sigma'], dtype=torch.float32
    )
    dataset = BCDataDataset(root=root_dir, split=split, target_transform=heatmap_gen, input_normalization=cfg['input_normalization'])
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    out_dir = run_dir / ('error_vis_' + split)
    out_dir.mkdir(exist_ok=True)


    kernel_size = cfg['kernel_size']
    threshold = cfg['threshold']
    merge_radius = cfg['merge_radius']
    matching_radius = cfg['matching_radius']

    def scatter(ax, pts, color, label):
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=10, label=label, linewidths=0)

    with torch.no_grad():
        for idx, (img_batch, _, pos_pts_tuple, neg_pts_tuple) in enumerate(loader):
            if n_images is not None and idx >= n_images:
                break

            img_batch = img_batch.to(device)
            logits = model(img_batch)
            pred_heatmaps = torch.sigmoid(logits)

            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=pred_heatmaps,
                kernel_size=kernel_size,
                threshold=threshold,
                merge_radius=merge_radius
            )

            pred_pos = out_pos[0]
            pred_neg = out_neg[0]
            gt_pos = pos_pts_tuple[0].astype(np.float32)
            gt_neg = neg_pts_tuple[0].astype(np.float32)

            tp_pos, fp_pos, fn_pos = match_points(pred_pos, gt_pos, matching_radius)
            tp_neg, fp_neg, fn_neg = match_points(pred_neg, gt_neg, matching_radius)

            # get original image as numpy for plotting
            img_np = img_batch[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            if cfg['input_normalization'] == 'imagenet':
                img_np = img_np * np.array(IMAGENET_STD).reshape(1,1,3) + np.array(IMAGENET_MEAN).reshape(1,1,3)
            img_np = np.clip(img_np, 0.0, 1.0)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            ax1.imshow(img_np)
            scatter(ax1, tp_pos, 'lime',  'TP')
            scatter(ax1, fp_pos, 'red',   'FP')
            scatter(ax1, fn_pos, 'blue',  'FN')
            ax1.set_title(f'Positive cells | TP={len(tp_pos)} FP={len(fp_pos)} FN={len(fn_pos)}')
            ax1.legend()
            ax1.axis('off')

            ax2.imshow(img_np)
            scatter(ax2, tp_neg, 'lime',  'TP')
            scatter(ax2, fp_neg, 'red',   'FP')
            scatter(ax2, fn_neg, 'blue',  'FN')
            ax2.set_title(f'Negative cells | TP={len(tp_neg)} FP={len(fp_neg)} FN={len(fn_neg)}')
            ax2.legend()
            ax2.axis('off')

            fig.savefig(out_dir / f'image_{idx:03d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved image_{idx:03d}.png")

            fig_gt, (ax_gt1, ax_gt2) = plt.subplots(1, 2, figsize=(16, 8))

            ax_gt1.imshow(img_np)
            scatter(ax_gt1, gt_pos, 'lime', 'GT pos')
            ax_gt1.set_title(f'GT Positive cells | n={len(gt_pos)}')
            ax_gt1.legend()
            ax_gt1.axis('off')

            ax_gt2.imshow(img_np)
            scatter(ax_gt2, gt_neg, 'blue', 'GT neg')
            ax_gt2.set_title(f'GT Negative cells | n={len(gt_neg)}')
            ax_gt2.legend()
            ax_gt2.axis('off')

            fig_gt.savefig(out_dir / f'image_{idx:03d}_gt.png', dpi=150, bbox_inches='tight')
            plt.close(fig_gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--n_images', type=int, default=None)
    parser.add_argument('--split', default='test', choices=['train', 'test', 'validation'])
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    main(Path(args.run_dir), args.n_images, args.split, args.checkpoint)