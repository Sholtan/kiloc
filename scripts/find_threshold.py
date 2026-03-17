import yaml, json, torch
import numpy as np
from pathlib import Path
import argparse
from torch.utils.data import DataLoader

from kiloc.utils.config import get_paths
from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.evaluation.metrics import compute_metrics


def main(run_dir, split, checkpoint, thresholds):
    with open(run_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)


    # ckpt_paths = list(run_dir.glob('*.pth'))
    # assert len(ckpt_paths) == 1, f"Expected 1 checkpoint, found {ckpt_paths}"
    if checkpoint is not None:
        print("checkpoint is not none")
        ckpt_paths = [run_dir / checkpoint]
    else:
        print("checkpoint is none")
        ckpt_paths = list(run_dir.glob('*.pth'))
        assert len(ckpt_paths) == 1, f"Expected 1 checkpoint, found {ckpt_paths}"

    # def model
    model = KiLocNet(pretrained=False)
    model.load_state_dict(torch.load(ckpt_paths[0], map_location='cpu'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # dataset and dataloader
    root_dir, _ = get_paths(device='h200')
    heatmap_gen = LocHeatmap(out_hw=cfg['out_hw'], in_hw=cfg['in_hw'],
                            sigma=cfg['sigma'], dtype=torch.float32)
    dataset = BCDataDataset(root=root_dir, split=split, target_transform=heatmap_gen)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    print(f"Running inference on {split} set...")
    cache = []  # list of (heatmap_tensor, gt_pos, gt_neg)

    with torch.no_grad():
        for img_batch, _, pos_pts_tuple, neg_pts_tuple in loader:
            img_batch = img_batch.to(device)
            logits = model(img_batch)
            heatmap = torch.sigmoid(logits[0]).cpu()  # (2, H, W)
            gt_pos = pos_pts_tuple[0]
            gt_neg = neg_pts_tuple[0]
            cache.append((heatmap, gt_pos, gt_neg))

    print(f"Cached {len(cache)} images. Sweeping thresholds...")


    results = []

    for threshold in thresholds:
        all_tp, all_fp, all_fn = 0, 0, 0

        for heatmap, gt_pos, gt_neg in cache:
            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=heatmap.unsqueeze(0),
                kernel_size=cfg['kernel_size'],
                threshold=threshold,
                merge_radius=cfg['merge_radius']
            )
            tp, fp, fn = compute_metrics(out_pos[0], gt_pos, cfg['matching_radius'])
            all_tp += tp; all_fp += fp; all_fn += fn

            tp, fp, fn = compute_metrics(out_neg[0], gt_neg, cfg['matching_radius'])
            all_tp += tp; all_fp += fp; all_fn += fn


        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1})
        print(f"threshold={threshold:.3f} | P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    results.sort(key=lambda x: x['f1'], reverse=True)
    best = results[0]
    print(f"\nBest threshold: {best['threshold']} | F1={best['f1']:.4f} P={best['precision']:.3f} R={best['recall']:.3f}")

    output = {
        'run_dir': str(run_dir),
        'split': split,
        'checkpoint': str(ckpt_paths[0].name),
        'thresholds_swept': thresholds,
        'results': results,
    }
    with open(run_dir / 'threshold_search.json', 'w') as f:
        json.dump(output, f, indent=2)
        json.dump(results, f, indent=2)
    print(f"Saved to {run_dir / 'threshold_search.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'test', 'validation'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[round(0.5 + i*0.025, 3) for i in range (20)])
    args = parser.parse_args()
    main(Path(args.run_dir), args.split, args.checkpoint, args.thresholds)