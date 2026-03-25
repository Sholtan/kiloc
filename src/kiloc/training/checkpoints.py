from __future__ import annotations

from pathlib import Path

import torch


def build_best_checkpoint_name(*, best_epoch: int, use_ema: bool) -> str:
    suffix = "_ema" if use_ema else ""
    return f"kilocnet_best_f1_epoch_{best_epoch}{suffix}.pth"


def save_single_best_checkpoint(
    *,
    run_dir: str | Path,
    best_epoch: int,
    model: torch.nn.Module,
    ema_module: torch.nn.Module | None = None,
    previous_checkpoint_path: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / build_best_checkpoint_name(
        best_epoch=best_epoch,
        use_ema=ema_module is not None,
    )

    if previous_checkpoint_path is not None:
        previous_checkpoint_path = Path(previous_checkpoint_path)
        if previous_checkpoint_path.exists() and previous_checkpoint_path != checkpoint_path:
            previous_checkpoint_path.unlink()

    state_dict = ema_module.state_dict() if ema_module is not None else model.state_dict()
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path
