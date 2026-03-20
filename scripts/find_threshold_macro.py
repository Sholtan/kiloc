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
from kiloc.evaluation.tta import tta_forward

def safe_div(num, den):
    return num / den if den > 0 else 0.0


def safe_f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def main(run_dir, split, checkpoint, thresholds):
    with open(run_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    use_tta = cfg.get('tta', False)
    print(f"use_tta: {use_tta}")
    if checkpoint is not None:
        print("checkpoint is not none")
        ckpt_paths = [run_dir / checkpoint]
    else:
        print("checkpoint is none")
        ckpt_paths = list(run_dir.glob('*.pth'))
        assert len(ckpt_paths) == 1, f"Expected 1 checkpoint, found {ckpt_paths}"

    # model
    model = KiLocNet(pretrained=False)
    model.load_state_dict(torch.load(ckpt_paths[0], map_location='cpu'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # dataset and dataloader
    root_dir, _ = get_paths(device='h200')
    heatmap_gen = LocHeatmap(
        out_hw=cfg['out_hw'],
        in_hw=cfg['in_hw'],
        sigma=cfg['sigma'],
        dtype=torch.float32
    )
    dataset = BCDataDataset(root=root_dir, split=split, target_transform=heatmap_gen, input_normalization=cfg['input_normalization'])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    print(f"Running inference on {split} set...")
    cache = []  # list of (heatmap_tensor, gt_pos, gt_neg)

    with torch.no_grad():
        for img_batch, _, pos_pts_tuple, neg_pts_tuple in loader:
            img_batch = img_batch.to(device)
            

            #logits = model(img_batch)
            #heatmap = torch.sigmoid(logits[0]).cpu()  # (2, H, W)
            if use_tta:
                heatmap = tta_forward(model, img_batch)[0].cpu()
            else:
                heatmap = torch.sigmoid(model(img_batch)[0]).cpu()

            gt_pos = pos_pts_tuple[0]
            gt_neg = neg_pts_tuple[0]
            cache.append((heatmap, gt_pos, gt_neg))

    print(f"Cached {len(cache)} images. Sweeping thresholds...")

    results = []

    for threshold in thresholds:
        # keep separate counts for positive and negative classes
        tp_pos_all, fp_pos_all, fn_pos_all = 0, 0, 0
        tp_neg_all, fp_neg_all, fn_neg_all = 0, 0, 0

        for heatmap, gt_pos, gt_neg in cache:
            out_pos, out_neg = heatmaps_to_points_batch(
                heatmaps=heatmap.unsqueeze(0),
                kernel_size=cfg['kernel_size'],
                threshold=threshold,
                merge_radius=cfg['merge_radius']
            )

            tp, fp, fn = compute_metrics(out_pos[0], gt_pos, cfg['matching_radius'])
            tp_pos_all += tp
            fp_pos_all += fp
            fn_pos_all += fn

            tp, fp, fn = compute_metrics(out_neg[0], gt_neg, cfg['matching_radius'])
            tp_neg_all += tp
            fp_neg_all += fp
            fn_neg_all += fn

        # positive metrics
        precision_pos = safe_div(tp_pos_all, tp_pos_all + fp_pos_all)
        recall_pos = safe_div(tp_pos_all, tp_pos_all + fn_pos_all)
        f1_pos = safe_f1(precision_pos, recall_pos)

        # negative metrics
        precision_neg = safe_div(tp_neg_all, tp_neg_all + fp_neg_all)
        recall_neg = safe_div(tp_neg_all, tp_neg_all + fn_neg_all)
        f1_neg = safe_f1(precision_neg, recall_neg)

        # macro metrics
        precision_macro = 0.5 * (precision_pos + precision_neg)
        recall_macro = 0.5 * (recall_pos + recall_neg)
        f1_macro = 0.5 * (f1_pos + f1_neg)

        # micro / pooled metrics
        all_tp = tp_pos_all + tp_neg_all
        all_fp = fp_pos_all + fp_neg_all
        all_fn = fn_pos_all + fn_neg_all

        precision_micro = safe_div(all_tp, all_tp + all_fp)
        recall_micro = safe_div(all_tp, all_tp + all_fn)
        f1_micro = safe_f1(precision_micro, recall_micro)

        result = {
            'threshold': threshold,

            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,

            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,

            'precision_pos': precision_pos,
            'recall_pos': recall_pos,
            'f1_pos': f1_pos,

            'precision_neg': precision_neg,
            'recall_neg': recall_neg,
            'f1_neg': f1_neg,

            'tp_pos': tp_pos_all,
            'fp_pos': fp_pos_all,
            'fn_pos': fn_pos_all,

            'tp_neg': tp_neg_all,
            'fp_neg': fp_neg_all,
            'fn_neg': fn_neg_all,
        }
        results.append(result)

        print(
            f"threshold={threshold:.3f} | "
            f"micro: P={precision_micro:.3f} R={recall_micro:.3f} F1={f1_micro:.3f} | "
            f"macro: P={precision_macro:.3f} R={recall_macro:.3f} F1={f1_macro:.3f} | "
            f"pos F1={f1_pos:.3f} neg F1={f1_neg:.3f}"
        )

    # choose best threshold by macro F1
    results.sort(key=lambda x: x['f1_macro'], reverse=True)
    best = results[0]

    print(
        f"\nBest threshold: {best['threshold']} | "
        f"macro F1={best['f1_macro']:.4f} "
        f"(micro F1={best['f1_micro']:.4f}, "
        f"pos F1={best['f1_pos']:.4f}, neg F1={best['f1_neg']:.4f})"
    )

    output = {
        'run_dir': str(run_dir),
        'split': split,
        'checkpoint': str(ckpt_paths[0].name),
        'thresholds_swept': thresholds,
        'best_threshold_by': 'f1_macro',
        'results': results,
    }

    with open(run_dir / f'threshold_search_macro_{checkpoint}.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {run_dir / 'threshold_search_macro.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'test', 'validation'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        #default=[round(0.5 + i * 0.025, 3) for i in range(20)]
        #default=[round(0.025 + i * 0.025, 3) for i in range(20)]
        default=[round(0.025 + i * 0.025, 3) for i in range(39)]
    )
    args = parser.parse_args()
    main(Path(args.run_dir), args.split, args.checkpoint, args.thresholds)