import yaml, json, torch
from pathlib import Path
import argparse
from torch.utils.data import DataLoader

from kiloc.utils.config import get_paths
from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.training.train import val_one_epoch

from kiloc.losses.losses import SigmoidWeightedMSE


def main(run_dir, split, checkpoint, r_interclass):
    with open(run_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)

    if checkpoint is not None:
        ckpt_paths = [run_dir / checkpoint]
    else:
        ckpt_paths = list(run_dir.glob('*.pth'))
        assert len(ckpt_paths) == 1, f"Expected 1 checkpoint, found {ckpt_paths}"

    model = KiLocNet(pretrained=False, backbone_name=cfg['backbone'])
    model.load_state_dict(torch.load(ckpt_paths[0], map_location='cpu'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    root_dir, _ = get_paths(device='h200')

    heatmap_gen = LocHeatmap(
        out_hw=cfg['out_hw'], in_hw=cfg['in_hw'],
        sigma=cfg['sigma'], dtype=torch.float32
    )
    dataset = BCDataDataset(root=root_dir, split=split, target_transform=heatmap_gen, input_normalization=cfg['input_normalization'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)


    if cfg['loss'] == 'sigmoid_weighted_mse_loss':
        criterion = SigmoidWeightedMSE(alpha_pos=cfg['alpha_pos'], alpha_neg=cfg['alpha_neg'], q=cfg['q'])
    elif cfg['loss'] == 'sigmoid_focal_loss':
        from kiloc.losses.losses import sigmoid_focal_loss
        criterion = sigmoid_focal_loss
    else:
        raise ValueError(f"Unknown loss: {cfg['loss']}")

    # 
    threshold=(cfg['thr_pos'], cfg['thr_neg'])

    val_loss, precision, recall, f1, \
        precision_pos, recall_pos, f1_pos, \
        precision_neg, recall_neg, f1_neg, f1_macro = val_one_epoch(
            model=model, criterion=criterion, device=device, val_loader=loader,
            kernel_size=cfg['kernel_size'], threshold = threshold,
            merge_radius=cfg['merge_radius'], matching_radius=cfg['matching_radius'],
            tta=cfg.get('tta', False), r_interclass = r_interclass
        )

    results = {
        'split': split,
        'val_loss': val_loss,
        'precision': precision, 'recall': recall, 'f1': f1,
        'precision_pos': precision_pos, 'recall_pos': recall_pos, 'f1_pos': f1_pos,
        'precision_neg': precision_neg, 'recall_neg': recall_neg, 'f1_neg': f1_neg,
        'f1_macro': f1_macro
    }

    for k, v in results.items():
        print(f"{k}: {v}")

    res_name = split + '_results.json'
    with open(run_dir / res_name, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {run_dir / res_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--split', default='validation', choices=['train', 'test', 'validation'])
    parser.add_argument('--checkpoint', default = None)
    parser.add_argument('--r_interclass', default=15., type=float)
    args = parser.parse_args()
    main(Path(args.run_dir), args.split, args.checkpoint, args.r_interclass)