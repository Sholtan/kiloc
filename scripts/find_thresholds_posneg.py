import yaml, json, torch
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
import numpy as np

from kiloc.utils.config import get_paths
from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.evaluation.decode import heatmaps_to_points_batch
from kiloc.evaluation.metrics import compute_metrics


def safe_div(num, den):
    return num / den if den > 0 else 0.0


def safe_f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def main(run_dir, split, checkpoint, thresholds_pos, thresholds_neg):
    with open(run_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)

    if checkpoint is not None:
        ckpt_paths = [run_dir / checkpoint]
    else:
        ckpt_paths = list(run_dir.glob('*.pth'))
        assert len(ckpt_paths) == 1, f"Expected 1 checkpoint, found {ckpt_paths}"

    model = KiLocNet(pretrained=False)
    model.load_state_dict(torch.load(ckpt_paths[0], map_location='cpu'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    root_dir, _ = get_paths(device='h200')
    heatmap_gen = LocHeatmap(
        out_hw=cfg['out_hw'],
        in_hw=cfg['in_hw'],
        sigma=cfg['sigma'],
        dtype=torch.float32
    )

    dataset = BCDataDataset(
        root=root_dir,
        split=split,
        target_transform=heatmap_gen,
        input_normalization=cfg.get('input_normalization', 'none'),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    print(f"Running inference on {split} set...")
    cache = []

    with torch.no_grad():
        for img_batch, _, pos_pts_tuple, neg_pts_tuple in loader:
            img_batch = img_batch.to(device)
            logits = model(img_batch)
            heatmap = torch.sigmoid(logits[0]).cpu()
            gt_pos = pos_pts_tuple[0]
            gt_neg = neg_pts_tuple[0]
            cache.append((heatmap, gt_pos, gt_neg))

    print(f"Cached {len(cache)} images. Sweeping thresholds...")
    results = []

    for thr_pos in thresholds_pos:
        for thr_neg in thresholds_neg:
            tp_pos_all = fp_pos_all = fn_pos_all = 0
            tp_neg_all = fp_neg_all = fn_neg_all = 0

            for heatmap, gt_pos, gt_neg in cache:
                out_pos, out_neg = heatmaps_to_points_batch(
                    heatmaps=heatmap.unsqueeze(0),
                    kernel_size=cfg['kernel_size'],
                    threshold=(thr_pos, thr_neg),
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

            precision_pos = safe_div(tp_pos_all, tp_pos_all + fp_pos_all)
            recall_pos = safe_div(tp_pos_all, tp_pos_all + fn_pos_all)
            f1_pos = safe_f1(precision_pos, recall_pos)

            precision_neg = safe_div(tp_neg_all, tp_neg_all + fp_neg_all)
            recall_neg = safe_div(tp_neg_all, tp_neg_all + fn_neg_all)
            f1_neg = safe_f1(precision_neg, recall_neg)

            precision_macro = 0.5 * (precision_pos + precision_neg)
            recall_macro = 0.5 * (recall_pos + recall_neg)
            f1_macro = 0.5 * (f1_pos + f1_neg)

            all_tp = tp_pos_all + tp_neg_all
            all_fp = fp_pos_all + fp_neg_all
            all_fn = fn_pos_all + fn_neg_all

            precision_micro = safe_div(all_tp, all_tp + all_fp)
            recall_micro = safe_div(all_tp, all_tp + all_fn)
            f1_micro = safe_f1(precision_micro, recall_micro)

            result = {
                'thr_pos': thr_pos,
                'thr_neg': thr_neg,

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
                f"thr_pos={thr_pos:.3f} thr_neg={thr_neg:.3f} | "
                f"micro: P={precision_micro:.3f} R={recall_micro:.3f} F1={f1_micro:.3f} | "
                f"macro: P={precision_macro:.3f} R={recall_macro:.3f} F1={f1_macro:.3f} | "
                f"pos F1={f1_pos:.3f} neg F1={f1_neg:.3f}"
            )

    results.sort(key=lambda x: x['f1_macro'], reverse=True)
    best = results[0]

    print(
        f"\nBest thresholds: thr_pos={best['thr_pos']:.3f}, thr_neg={best['thr_neg']:.3f} | "
        f"macro F1={best['f1_macro']:.4f} "
        f"(micro F1={best['f1_micro']:.4f}, "
        f"pos F1={best['f1_pos']:.4f}, neg F1={best['f1_neg']:.4f})"
    )

    output = {
        'run_dir': str(run_dir),
        'split': split,
        'checkpoint': str(ckpt_paths[0].name),
        'thresholds_pos_swept': thresholds_pos,
        'thresholds_neg_swept': thresholds_neg,
        'best_threshold_by': 'f1_macro',
        'results': results,
    }

    with open(run_dir / 'threshold_search_macro_2d.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {run_dir / 'threshold_search_macro_2d.json'}")


def expand_sweep(spec, name: str):
    """
    spec: [start, end, num_values]
    Returns a Python list including both start and end.
    Example: [0.7, 0.875, 8] -> 8 evenly spaced values.
    """
    if len(spec) != 3:
        raise ValueError(
            f"{name} must have exactly 3 values: start end num_values. Got: {spec}"
        )

    start, end, num = spec
    num = int(num)

    if num < 2:
        raise ValueError(f"{name}: num_values must be >= 2, got {num}")
    if end < start:
        raise ValueError(f"{name}: end must be >= start, got start={start}, end={end}")

    values = np.linspace(float(start), float(end), num=num, endpoint=True)
    return [round(float(x), 6) for x in values]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'test', 'validation'])
    parser.add_argument('--checkpoint', default=None)

    parser.add_argument(
        '--thresholds_pos',
        nargs=3,
        type=float,
        metavar=('START', 'END', 'NUM'),
        default=[0.70, 0.875, 8],
        help='Positive threshold sweep as: START END NUM (inclusive)'
    )
    parser.add_argument(
        '--thresholds_neg',
        nargs=3,
        type=float,
        metavar=('START', 'END', 'NUM'),
        default=[0.55, 0.775, 10],
        help='Negative threshold sweep as: START END NUM (inclusive)'
    )

    args = parser.parse_args()

    thresholds_pos = expand_sweep(args.thresholds_pos, 'thresholds_pos')
    thresholds_neg = expand_sweep(args.thresholds_neg, 'thresholds_neg')

    print('thresholds_pos =', thresholds_pos)
    print('thresholds_neg =', thresholds_neg)

    main(
        Path(args.run_dir),
        args.split,
        args.checkpoint,
        thresholds_pos,
        thresholds_neg,
    )