"""
Runs the full train/val loop
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from kiloc.utils.config import get_paths
from kiloc.datasets.bcdata import BCDataDataset, collate_fn
from kiloc.target_generation.heatmaps import LocHeatmap
from kiloc.model.kiloc_net import KiLocNet
from kiloc.losses.losses import sigmoid_weighted_mse_loss, sigmoid_focal_loss
from kiloc.training.train import train_one_epoch, val_one_epoch


def main():
    # get the paths
    root_dir, checkpoint_dir = get_paths(device='hpvictus')

    # build dataloaders
    heatmap_gen = LocHeatmap(out_hw=(160, 160), in_hw=(
        640, 640), sigma=3., dtype=torch.float32)
    dataset_train = BCDataDataset(
        root=root_dir, split='train', target_transform=heatmap_gen)
    dataset_val = BCDataDataset(
        root=root_dir, split='test', target_transform=heatmap_gen)
    batch_size = 2
    num_workers = 4
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=True, drop_last=True, persistent_workers=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=True, drop_last=True, persistent_workers=True)

    # build model
    model = KiLocNet(pretrained=True)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = model.to(device)
    # build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

    # Loss function
    # criterion = sigmoid_focal_loss
    criterion = sigmoid_weighted_mse_loss

    # evaluation settings:
    kernel_size = 3
    threshold = 0.5
    merge_radius = 1.5
    matching_radius = 10

    epochs = 1

    loss_arr_train = []
    loss_arr_val = []

    metrics_history = []

    best_val_loss = np.inf

    for i in range(epochs):
        total_loss_train = train_one_epoch(model=model, criterion=criterion,
                                           optimizer=optimizer, device=device,
                                           trainloader=dataloader_train)
        loss_arr_train.append(total_loss_train)

        metrics = {}
        total_loss_val, precision, recall, f1 = val_one_epoch(model=model, criterion=criterion,
                                                              device=device, val_loader=dataloader_val,
                                                              kernel_size=kernel_size, threshold=threshold,
                                                              merge_radius=merge_radius, matching_radius=matching_radius)
        print(f"Epoch {i+1}/{epochs} | train={total_loss_train:.4f} | val={total_loss_val:.4f} | P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
        loss_arr_val.append(total_loss_val)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics_history.append(metrics)

        if total_loss_val < best_val_loss:
            best_val_loss = total_loss_val
            checkpoint_path = checkpoint_dir / "kilocnet_v0_best.pth"
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()
