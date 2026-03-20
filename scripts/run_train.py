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
import random
import os

from kiloc.utils.config import get_paths
from kiloc.datasets.bcdata import BCDataDataset, collate_fn, AlbumentationsJointTransform
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.training.train import train_one_epoch, val_one_epoch
from kiloc.training.ema import ModelEMA
from kiloc.losses.losses import sigmoid_focal_loss, SigmoidWeightedMSE, SigmoidSumHuber

import albumentations as A

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # This disables cuDNN's non-deterministic algorithms but can slow down training slightly. Whether you need it depends on how strict reproducibility needs to be.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# for the dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(config_path, run_suffix, out_dir):
    # config parameters:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    epochs = cfg['epochs']

    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    lr = cfg['lr']
    weight_decay = cfg['weight_decay']
    is_pretrained = cfg['is_pretrained']
    input_normalization = cfg['input_normalization']
    sigma = cfg['sigma']
    out_hw = cfg['out_hw']
    in_hw = cfg['in_hw']

    # SEED
    set_seed(cfg['seed'])

    


    # Loss function
    if cfg['loss'] == 'sigmoid_weighted_mse_loss':
        detection_loss = SigmoidWeightedMSE(alpha_pos = cfg['alpha_pos'], alpha_neg = cfg['alpha_neg'], q = cfg['q'])
    elif cfg['loss'] == 'sigmoid_focal_loss':
        detection_loss = sigmoid_focal_loss
    else:
        raise ValueError(f"must choose one of losses: sigmoid_weighted_mse_loss or sigmoid_focal_loss, \
                         got {cfg['loss']} instead")


    if cfg.get('count_loss', False):
        print("Building composite loss with addition of count(Huber) loss")
        sum_huber = SigmoidSumHuber()
        def criterion(pred, target, pos_pts_tuple, neg_pts_tuple):
            det = detection_loss(pred, target, pos_pts_tuple, neg_pts_tuple)
            cnt = sum_huber(pred, target, pos_pts_tuple, neg_pts_tuple)
            return det + cfg['lambda_count'] * cnt
    else:
        criterion = detection_loss
    

    # evaluation settings:
    kernel_size = cfg['kernel_size']

    base_threshold = cfg.get('threshold', 0.5)
    thr_pos = cfg.get('thr_pos', base_threshold)
    thr_neg = cfg.get('thr_neg', base_threshold)
    threshold = (thr_pos, thr_neg)
    print(f"Validation thresholds: thr_pos={thr_pos:.3f}, thr_neg={thr_neg:.3f}")
    
    merge_radius = cfg['merge_radius']
    matching_radius = cfg['matching_radius']


    # get the paths
    root_dir, checkpoint_dir = get_paths(device='h200')

    if out_dir:
        checkpoint_dir = checkpoint_dir / out_dir
    
    # create run's save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_suffix:
        run_dir = checkpoint_dir / ('run_' + timestamp  + '_' + run_suffix)
    else:
        run_dir = checkpoint_dir / ('run_' + timestamp)
    run_dir.mkdir(parents = True)


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


    
    heatmap_gen = LocHeatmap(out_hw = out_hw, in_hw=in_hw, 
                             sigma=sigma, dtype=torch.float32)
    
    # build datasets
    joint_tf = train_tf if cfg.get('augmentation', False) else None
    if joint_tf:
        print('Augmentations will be applied to the train set')
    else:
        print("No augmentations will be applied")

    print(f"Using {input_normalization} input normalization")
    dataset_train = BCDataDataset(
        root=root_dir,
        split='train',
        target_transform=heatmap_gen,
        joint_transform=joint_tf,
        input_normalization=input_normalization,
    )

    dataset_val = BCDataDataset(
        root=root_dir,
        split='validation',
        target_transform=heatmap_gen,
        input_normalization=input_normalization,
    )
    
    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    # build dataloaders
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  worker_init_fn=seed_worker, collate_fn=collate_fn, pin_memory=True, drop_last=False, generator=g, persistent_workers=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                worker_init_fn=seed_worker, collate_fn=collate_fn, pin_memory=True, drop_last=False, generator=g, persistent_workers=True)

    # build model
    model = KiLocNet(pretrained=is_pretrained)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = model.to(device)
    use_ema = cfg.get("ema", False)
    ema = None
    ema_start_epoch = cfg.get("ema_start_epoch", 5)

    # build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr,
        weight_decay = weight_decay,
    )


    # build scheduler
    sch_cfg = cfg["scheduler"]
    if sch_cfg["name"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode=sch_cfg["mode"], 
                                                               factor=sch_cfg["factor"], patience=sch_cfg["patience"], 
                                                               min_lr=sch_cfg["min_lr"])
    elif sch_cfg["name"] == "none":
        scheduler = None
    else:
        raise ValueError(f"scheduler must be ReduceLROnPlateau, it's only one implemented yet")
    
    #best_val_loss = np.inf
    best_f1_macro = -1.
    history = []
    best_epoch = -1
    for i in range(epochs):
        if use_ema and ema is None and i == ema_start_epoch:
            ema = ModelEMA(
                model=model,                     # important: copy CURRENT trained model
                decay=cfg.get("ema_decay", 0.999),
                device=cfg.get("ema_device", None),
            )
            print(f"EMA initialized at epoch {i+1}")
            
        total_loss_train = train_one_epoch(model=model, criterion=criterion,
                                           optimizer=optimizer, device=device,
                                           trainloader=dataloader_train, ema=ema,)

        eval_model = ema.module if ema is not None else model
        val_result = val_one_epoch(model=eval_model, criterion=criterion,
                                                              device=device, val_loader=dataloader_val,
                                                              kernel_size=kernel_size, threshold=threshold,
                                                              merge_radius=merge_radius, matching_radius=matching_radius,
                                                              tta=False)
        total_loss_val, precision, recall, f1, \
            precision_pos, recall_pos, f1_pos, \
                precision_neg, recall_neg, f1_neg, f1_macro = val_result
        
        print(f"Epoch {i+1}/{epochs} | train={total_loss_train:.4f} | val={total_loss_val:.4f} | P={precision:.3f},  R={recall:.3f},  F1_micro={f1:.3f},  F1_macro:{f1_macro:.3f}")


        # save weights if val_los is new minimum
        # trying to save best f1
        if f1_macro > best_f1_macro:  #total_loss_val < best_val_loss:
            best_f1_macro = f1_macro
            torch.save(model.state_dict(), run_dir / "kilocnet_epoch_best.pth")
            best_epoch = i + 1
            if ema is not None:
                torch.save(ema.module.state_dict(), run_dir / "kilocnet_epoch_best_ema.pth")
        
        if i == epochs-1:
            torch.save(model.state_dict(), run_dir / "kilocnet_epoch_last.pth")
            if ema is not None:
                torch.save(ema.module.state_dict(), run_dir / "kilocnet_epoch_last_ema.pth")

        history.append({
            "epoch": i+1,
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
            "f1_macro": f1_macro,
        })
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        if scheduler is not None:
            scheduler.step(f1_macro)
        


    
    best_path = run_dir / "kilocnet_epoch_best.pth"
    best_path.rename(run_dir / f"kilocnet_best_f1_epoch_{best_epoch}.pth")
    
    best_path_ema = run_dir / "kilocnet_epoch_best_ema.pth"
    best_path_ema.rename(run_dir / f"kilocnet_best_f1_epoch_{best_epoch}_ema.pth")

    print(f"BEST EPOCH WAS: {best_epoch}")

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_1.yaml')
    parser.add_argument('--run_suffix', default=None)
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()
    main(args.config, args.run_suffix, args.out_dir)
