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
from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.losses.losses import sigmoid_weighted_mse_loss, sigmoid_focal_loss
from kiloc.training.train import train_one_epoch, val_one_epoch


def main(config_path):
    # config parameters:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Loss function
    if cfg['loss'] == 'sigmoid_weighted_mse_loss':
        criterion = sigmoid_weighted_mse_loss
    elif cfg['loss'] == 'sigmoid_focal_loss':
        criterion = sigmoid_focal_loss
    else:
        raise ValueError(f"must choose one of losses: sigmoid_weighted_mse_loss or sigmoid_focal_loss, \
                         got {cfg['loss']} instead")

    epochs = cfg['epochs']

    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    lr = cfg['lr']
    weight_decay = cfg['weight_decay']
    is_pretrained = cfg['is_pretrained']
    sigma = cfg['sigma']
    out_hw = cfg['out_hw']
    in_hw = cfg['in_hw']

    

    # evaluation settings:
    kernel_size = cfg['kernel_size']
    threshold = cfg['threshold']
    merge_radius = cfg['merge_radius']
    matching_radius = cfg['matching_radius']


    # get the paths
    root_dir, checkpoint_dir = get_paths(device='h200')
    
    # create run's save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = checkpoint_dir / ('run_' + timestamp)
    run_dir.mkdir(parents = True)

    # save the config before run starts
    shutil.copy(config_path, run_dir / 'config.yaml')
    

    # build dataloaders
    heatmap_gen = LocHeatmap(out_hw = out_hw, in_hw=in_hw, 
                             sigma=sigma, dtype=torch.float32)
    dataset_train = BCDataDataset(
        root=root_dir, split='train', target_transform=heatmap_gen)
    dataset_val = BCDataDataset(
        root=root_dir, split='test', target_transform=heatmap_gen)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=True, drop_last=True, persistent_workers=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=True, drop_last=True, persistent_workers=True)

    # build model
    model = KiLocNet(pretrained=is_pretrained)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = model.to(device)
    
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
    best_f1 = -1.
    history = []
    best_epoch = -1
    for i in range(epochs):
        total_loss_train = train_one_epoch(model=model, criterion=criterion,
                                           optimizer=optimizer, device=device,
                                           trainloader=dataloader_train)

        val_result = val_one_epoch(model=model, criterion=criterion,
                                                              device=device, val_loader=dataloader_val,
                                                              kernel_size=kernel_size, threshold=threshold,
                                                              merge_radius=merge_radius, matching_radius=matching_radius)
        total_loss_val, precision, recall, f1, \
            precision_pos, recall_pos, f1_pos, \
                precision_neg, recall_neg, f1_neg = val_result
        
        print(f"Epoch {i+1}/{epochs} | train={total_loss_train:.4f} | val={total_loss_val:.4f} | P={precision:.3f} R={recall:.3f} F1={f1:.3f}")


        # save weights if val_los is new minimum
        # trying to save best f1
        if f1 > best_f1:  #total_loss_val < best_val_loss:
            #best_val_loss = total_loss_val
            best_f1 = f1
            torch.save(model.state_dict(), run_dir / "kilocnet_epoch_best.pth")
            best_epoch = i + 1

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
        })
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
    args = parser.parse_args()
    main(args.config)
