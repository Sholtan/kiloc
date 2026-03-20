from __future__ import annotations

from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential moving average of model parameters + buffers.

    - Train with the raw model.
    - After each optimizer.step(), call ema.update(model).
    - Validate / save ema.module.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9997,
        device: str | torch.device | None = None,
    ) -> None:
        self.module = deepcopy(model).eval()
        self.decay = decay
        self.device = device

        if self.device is not None:
            self.module.to(self.device)

        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()

        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()

            if self.device is not None:
                model_v = model_v.to(self.device)

            if not model_v.dtype.is_floating_point:
                # e.g. num_batches_tracked
                ema_v.copy_(model_v)
            else:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)



