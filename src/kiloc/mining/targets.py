from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HardNegativeWeightMapGenerator:
    out_hw: tuple[int, int]
    in_hw: tuple[int, int]
    sigma: float = 2.0
    dtype: torch.dtype = torch.float32
    aggregation: str = "max"
    use_score_weights: bool = False
    constant_weight: float = 1.0
    clamp_max: float | None = None

    def __call__(self, candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> torch.Tensor:
        out_h, out_w = self.out_hw
        weight_map = torch.zeros((2, out_h, out_w), dtype=self.dtype)

        if len(candidates) == 0:
            return weight_map

        if float(self.sigma) <= 0:
            raise ValueError("sigma must be > 0 for HardNegativeWeightMapGenerator")
        if self.aggregation not in {"max", "sum"}:
            raise ValueError("aggregation must be 'max' or 'sum'")

        in_h, in_w = self.in_hw
        sx = out_w / float(in_w)
        sy = out_h / float(in_h)

        yy = torch.arange(out_h, dtype=self.dtype).view(out_h, 1)
        xx = torch.arange(out_w, dtype=self.dtype).view(1, out_w)

        for row in candidates:
            pred_class = row["pred_class"]
            if pred_class not in {"pos", "neg"}:
                raise ValueError(f"pred_class must be 'pos' or 'neg', got {pred_class!r}")

            x = float(row["x"]) * sx
            y = float(row["y"]) * sy
            point_weight = float(row["score"]) if self.use_score_weights else float(self.constant_weight)

            gaussian = point_weight * torch.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * float(self.sigma) ** 2)
            )
            channel = 0 if pred_class == "pos" else 1

            if self.aggregation == "max":
                weight_map[channel] = torch.maximum(weight_map[channel], gaussian)
            else:
                weight_map[channel] += gaussian

        if self.clamp_max is not None:
            weight_map = weight_map.clamp_max(float(self.clamp_max))

        return weight_map
