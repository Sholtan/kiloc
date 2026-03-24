"""
Runs the full train/val loop
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from datetime import datetime
import argparse
import yaml
import shutil

from kiloc.utils.config import get_paths
from kiloc.datasets.bcdata import BCDataDataset, collate_fn, AlbumentationsJointTransform
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.training.train import val_one_epoch

from kiloc.losses.losses import sigmoid_focal_loss, SigmoidWeightedMSE, SigmoidSumHuber

import albumentations as A


def _unpack_batch(batch, device):
    """
    Expected batch format from collate_fn:
        images, target, pos_pts_tuple, neg_pts_tuple
    """
    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
        raise ValueError(
            "Expected batch to be a 4-tuple/list: "
            "(images, target, pos_pts_tuple, neg_pts_tuple)."
        )

    images, target, pos_pts_tuple, neg_pts_tuple = batch

    if torch.is_tensor(images):
        images = images.to(device, non_blocking=True)
    if torch.is_tensor(target):
        target = target.to(device, non_blocking=True)

    return images, target, pos_pts_tuple, neg_pts_tuple


def _grad_l2_norm(grads):
    sq_norm = None
    for g in grads:
        if g is None:
            continue
        term = g.detach().pow(2).sum()
        sq_norm = term if sq_norm is None else sq_norm + term

    if sq_norm is None:
        return 0.0
    return sq_norm.sqrt().item()


def _measure_loss_grad_norm(loss, params):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    return _grad_l2_norm(grads)


def _select_shared_params(model, scope='backbone_fpn'):
    """
    Select parameters on which to compare gradient norms of detection vs count loss.

    scope options:
      - all
      - backbone
      - backbone_fpn (default)
      - non_heads
    """
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        raise ValueError("Model has no trainable parameters.")

    scope = scope.lower()
    head_like_keys = ('head', 'cls', 'class', 'count', 'loc', 'detect', 'pred')

    if scope == 'all':
        selected = named_params
    elif scope == 'backbone':
        selected = [(n, p) for n, p in named_params if 'backbone' in n.lower()]
    elif scope == 'backbone_fpn':
        include_keys = ('backbone', 'fpn', 'neck', 'pyramid', 'lateral', 'topdown', 'stem', 'encoder', 'body')
        selected = [
            (n, p)
            for n, p in named_params
            if any(k in n.lower() for k in include_keys) and not any(k in n.lower() for k in head_like_keys)
        ]
        if not selected:
            # fallback: everything except obvious heads
            selected = [(n, p) for n, p in named_params if not any(k in n.lower() for k in head_like_keys)]
    elif scope == 'non_heads':
        selected = [(n, p) for n, p in named_params if not any(k in n.lower() for k in head_like_keys)]
    else:
        raise ValueError(f"Unknown grad-norm scope: {scope}")

    if not selected:
        selected = named_params
        print("[grad-norm] Warning: could not isolate shared trunk params; falling back to all trainable parameters.")

    selected_names = [n for n, _ in selected]
    selected_params = [p for _, p in selected]
    return selected_names, selected_params


class CompositeLoss:
    def __init__(self, detection_loss, count_loss=None, det_weight=1.0, cnt_weight=1.0):
        self.detection_loss = detection_loss
        self.count_loss = count_loss
        self.det_weight = det_weight
        self.cnt_weight = cnt_weight

    @property
    def has_count_loss(self):
        return self.count_loss is not None

    def compute_terms(self, pred, target, pos_pts_tuple, neg_pts_tuple):
        det = self.detection_loss(pred, target, pos_pts_tuple, neg_pts_tuple)
        if self.count_loss is None:
            cnt = None
            total = self.det_weight * det
        else:
            cnt = self.count_loss(pred, target, pos_pts_tuple, neg_pts_tuple)
            total = self.det_weight * det + self.cnt_weight * cnt
        return det, cnt, total

    def __call__(self, pred, target, pos_pts_tuple, neg_pts_tuple):
        _, _, total = self.compute_terms(pred, target, pos_pts_tuple, neg_pts_tuple)
        return total


@torch.no_grad()
def _safe_ratio(num, den):
    if den == 0.0:
        return None
    return num / den


def train_one_epoch_with_optional_grad_logging(
    model,
    optimizer,
    device,
    trainloader,
    loss_obj,
    measure_grad_norms=False,
    grad_norm_params=None,
    grad_every_n_steps=1,
    max_grad_batches=None,
):
    model.train()

    total_loss = 0.0
    total_det = 0.0
    total_cnt = 0.0
    n_batches = 0

    grad_det_total = 0.0
    grad_cnt_total = 0.0
    grad_ratio_total = 0.0
    grad_ratio_count = 0
    grad_batches = 0

    for step, batch in enumerate(trainloader):
        images, target, pos_pts_tuple, neg_pts_tuple = _unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        det, cnt, total = loss_obj.compute_terms(pred, target, pos_pts_tuple, neg_pts_tuple)

        do_measure = (
            measure_grad_norms
            and loss_obj.has_count_loss
            and grad_norm_params is not None
            and (step % grad_every_n_steps == 0)
            and (max_grad_batches is None or grad_batches < max_grad_batches)
        )

        if do_measure:
            det_grad_norm = _measure_loss_grad_norm(loss_obj.det_weight * det, grad_norm_params)
            cnt_grad_norm = _measure_loss_grad_norm(loss_obj.cnt_weight * cnt, grad_norm_params)
            grad_det_total += det_grad_norm
            grad_cnt_total += cnt_grad_norm
            ratio = _safe_ratio(cnt_grad_norm, det_grad_norm)
            if ratio is not None:
                grad_ratio_total += ratio
                grad_ratio_count += 1
            grad_batches += 1

        total.backward()
        optimizer.step()

        total_loss += total.item()
        total_det += det.item()
        if cnt is not None:
            total_cnt += cnt.item()
        n_batches += 1

    stats = {
        'train_det_loss': total_det / max(n_batches, 1),
        'train_cnt_loss': (total_cnt / max(n_batches, 1)) if loss_obj.has_count_loss else None,
        'grad_det_norm': (grad_det_total / grad_batches) if grad_batches > 0 else None,
        'grad_cnt_norm': (grad_cnt_total / grad_batches) if grad_batches > 0 else None,
        'grad_cnt_to_det_ratio': (grad_ratio_total / grad_ratio_count) if grad_ratio_count > 0 else None,
        'grad_batches_measured': grad_batches,
    }
    return total_loss / max(n_batches, 1), stats



def main(config_path, run_name):
    # config parameters:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    epochs = cfg['epochs']

    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    lr = cfg['lr']
    weight_decay = cfg['weight_decay']
    is_pretrained = cfg['is_pretrained']
    sigma = cfg['sigma']
    out_hw = cfg['out_hw']
    in_hw = cfg['in_hw']

    # optional loss weights
    det_weight = cfg.get('det_weight', 1.0)
    cnt_weight = cfg.get('cnt_weight', 1.0)

    # gradient logging config
    measure_grad_norms = cfg.get('measure_grad_norms', cfg.get('count_loss', False))
    measure_grad_epochs = cfg.get('measure_grad_epochs', [1])
    grad_norm_scope = cfg.get('grad_norm_scope', 'backbone_fpn')
    grad_every_n_steps = cfg.get('grad_norm_every_n_steps', 1)
    max_grad_batches_per_epoch = cfg.get('max_grad_batches_per_epoch', None)

    if measure_grad_epochs == 'all':
        measure_grad_epochs = None
    elif isinstance(measure_grad_epochs, int):
        measure_grad_epochs = {measure_grad_epochs}
    else:
        measure_grad_epochs = set(measure_grad_epochs)

    # Loss function
    if cfg['loss'] == 'sigmoid_weighted_mse_loss':
        detection_loss = SigmoidWeightedMSE(alpha_pos=cfg['alpha_pos'], alpha_neg=cfg['alpha_neg'], q=cfg['q'])
    elif cfg['loss'] == 'sigmoid_focal_loss':
        detection_loss = sigmoid_focal_loss
    else:
        raise ValueError(
            f"must choose one of losses: sigmoid_weighted_mse_loss or sigmoid_focal_loss, "
            f"got {cfg['loss']} instead"
        )

    if cfg.get('count_loss', False):
        print("Building composite loss with addition of count(Huber) loss")
        sum_huber = SigmoidSumHuber()
        loss_obj = CompositeLoss(
            detection_loss=detection_loss,
            count_loss=sum_huber,
            det_weight=det_weight,
            cnt_weight=cnt_weight,
        )
    else:
        loss_obj = CompositeLoss(
            detection_loss=detection_loss,
            count_loss=None,
            det_weight=det_weight,
            cnt_weight=cnt_weight,
        )

    criterion = loss_obj

    # evaluation settings:
    kernel_size = cfg['kernel_size']
    threshold = cfg['threshold']
    merge_radius = cfg['merge_radius']
    matching_radius = cfg['matching_radius']

    # get the paths
    root_dir, checkpoint_dir = get_paths(device='h200')

    # create run's save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_name:
        run_dir = checkpoint_dir / run_name
    else:
        run_dir = checkpoint_dir / ('run_' + timestamp)
    run_dir.mkdir(parents=True)

    print(f"RUN_DIR:{run_dir}")
    # save the config before run starts
    shutil.copy(config_path, run_dir / 'config.yaml')

    # augmentations
    train_tf = AlbumentationsJointTransform(A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.10, rotate_limit=0, border_mode=0, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=1.0),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=8, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.GaussNoise(std_range=(0.005, 0.015), mean_range=(0.0, 0.0), p=1.0),
            A.ImageCompression(quality_range=(85, 100), p=1.0),
        ], p=0.15),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)))

    heatmap_gen = LocHeatmap(out_hw=out_hw, in_hw=in_hw, sigma=sigma, dtype=torch.float32)

    # build datasets
    joint_tf = train_tf if cfg.get('augmentation', False) else None
    if joint_tf:
        print('Augmentations will be applied to the train set')
    else:
        print("No augmentations will be applied")
    dataset_train = BCDataDataset(
        root=root_dir, split='train', target_transform=heatmap_gen, joint_transform=joint_tf, input_normalization=cfg['input_normalization'])
    dataset_val = BCDataDataset(
        root=root_dir, split='validation', target_transform=heatmap_gen, input_normalization=cfg['input_normalization'])

    # build dataloaders
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # build model
    model = KiLocNet(pretrained=is_pretrained)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = model.to(device)

    grad_norm_names, grad_norm_params = _select_shared_params(model, scope=grad_norm_scope)
    print(
        f"[grad-norm] scope={grad_norm_scope} | selected_params={len(grad_norm_params)} | "
        f"sample={grad_norm_names[:5]}"
    )

    # build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # build scheduler
    sch_cfg = cfg["scheduler"]
    if sch_cfg["name"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=sch_cfg["mode"],
            factor=sch_cfg["factor"],
            patience=sch_cfg["patience"],
            min_lr=sch_cfg["min_lr"],
        )
    elif sch_cfg["name"] == "none":
        scheduler = None
    else:
        raise ValueError("scheduler must be ReduceLROnPlateau, it's only one implemented yet")

    #best_val_loss = np.inf
    best_f1 = -1.
    history = []
    best_epoch = -1
    for i in range(epochs):
        epoch_num = i + 1
        should_measure_this_epoch = (
            measure_grad_norms
            and loss_obj.has_count_loss
            and (measure_grad_epochs is None or epoch_num in measure_grad_epochs)
        )

        total_loss_train, train_stats = train_one_epoch_with_optional_grad_logging(
            model=model,
            loss_obj=loss_obj,
            optimizer=optimizer,
            device=device,
            trainloader=dataloader_train,
            measure_grad_norms=should_measure_this_epoch,
            grad_norm_params=grad_norm_params,
            grad_every_n_steps=grad_every_n_steps,
            max_grad_batches=max_grad_batches_per_epoch,
        )

        if loss_obj.has_count_loss:
            msg = (
                f"[epoch {epoch_num} train avg] "
                f"detection={train_stats['train_det_loss']:.6f}  "
                f"count={train_stats['train_cnt_loss']:.6f}"
            )
            if should_measure_this_epoch and train_stats['grad_det_norm'] is not None:
                msg += (
                    f"  | grad_det={train_stats['grad_det_norm']:.6f}"
                    f"  grad_cnt={train_stats['grad_cnt_norm']:.6f}"
                    f"  cnt/det={train_stats['grad_cnt_to_det_ratio']:.6f}"
                    f"  measured_batches={train_stats['grad_batches_measured']}"
                )
            print(msg)

        val_result = val_one_epoch(
            model=model,
            criterion=criterion,
            device=device,
            val_loader=dataloader_val,
            kernel_size=kernel_size,
            threshold=threshold,
            merge_radius=merge_radius,
            matching_radius=matching_radius,
        )
        total_loss_val, precision, recall, f1, \
            precision_pos, recall_pos, f1_pos, \
                precision_neg, recall_neg, f1_neg = val_result

        print(
            f"Epoch {epoch_num}/{epochs} | train={total_loss_train:.4f} | val={total_loss_val:.4f} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

        # save weights if val_los is new minimum
        # trying to save best f1
        if f1 > best_f1:  #total_loss_val < best_val_loss:
            #best_val_loss = total_loss_val
            best_f1 = f1
            torch.save(model.state_dict(), run_dir / "kilocnet_epoch_best.pth")
            best_epoch = epoch_num

        if i == epochs - 1:
            torch.save(model.state_dict(), run_dir / "kilocnet_epoch_last.pth")

        history_entry = {
            "epoch": epoch_num,
            "train_loss": total_loss_train,
            "val_loss": total_loss_val,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_pos": precision_pos,
            "recall_pos": recall_pos,
            "f1_pos": f1_pos,
            "precision_neg": precision_neg,
            "recall_neg": recall_neg,
            "f1_neg": f1_neg,
            "lr": optimizer.param_groups[0]['lr'],
            "train_det_loss": train_stats['train_det_loss'],
            "train_cnt_loss": train_stats['train_cnt_loss'],
            "grad_det_norm": train_stats['grad_det_norm'],
            "grad_cnt_norm": train_stats['grad_cnt_norm'],
            "grad_cnt_to_det_ratio": train_stats['grad_cnt_to_det_ratio'],
            "grad_batches_measured": train_stats['grad_batches_measured'],
        }
        history.append(history_entry)
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if scheduler is not None:
            scheduler.step(f1)

    best_path = run_dir / "kilocnet_epoch_best.pth"
    best_path.rename(run_dir / f"kilocnet_best_f1_epoch_{best_epoch}.pth")
    print(f"BEST EPOCH WAS: {best_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_1.yaml')
    parser.add_argument('--run_name', default=None)
    args = parser.parse_args()
    main(args.config, args.run_name)
