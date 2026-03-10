# Model definition: ResNet34 + FPN + HeatmapHead

## Date opened
2026-03-10

## Status
done

---

## Why this task exists

The dataset pipeline is validated. Before writing the training loop, the model must exist and be verified to produce correctly shaped output from a real input batch, with gradients flowing correctly.

---

## Task definition

Implement `KiLocNet` (backbone + FPN + localization heads) in `src/kiloc/model/`. Validate with `scripts/debug_model.py`.

---

## Desired end state

- `KiLocNet(pretrained=False).forward(x)` produces `(B, 2, 160, 160)` from `(B, 3, 640, 640)` input
- No NaN/Inf in output
- Backward pass runs without error
- Forward pass on a real DataLoader batch passes

---

## Relevant files

- `src/kiloc/model/__init__.py`
- `src/kiloc/model/backbone.py` — `build_resnet34_backbone(pretrained)`
- `src/kiloc/model/fpn.py` — `FPN(in_channels, out_channels)`
- `src/kiloc/model/head.py` — `HeatmapHead(in_channels, num_convs)`
- `src/kiloc/model/kiloc_net.py` — `KiLocNet(pretrained)`
- `scripts/debug_model.py`

---

## Architecture

```
input (B, 3, 640, 640)
→ ResNet34 backbone (torchvision, create_feature_extractor)
    c2: (B, 64,  160, 160)   stride 4
    c3: (B, 128,  80,  80)   stride 8
    c4: (B, 256,  40,  40)   stride 16
    c5: (B, 512,  20,  20)   stride 32
→ FPN (lateral 1×1 + top-down + 3×3 output convs, out_channels=256)
    p2: (B, 256, 160, 160)   ← only this level used
→ pos_head (HeatmapHead): (B, 1, 160, 160)
→ neg_head (HeatmapHead): (B, 1, 160, 160)
→ torch.cat → loc_hm: (B, 2, 160, 160)
```

`HeatmapHead`: 3× Conv3×3+ReLU followed by 1×1 conv, no final activation. Outputs raw logits.

---

## Key design decisions

- `build_resnet34_backbone` is a factory function, not a module-level instance (prevents import-time weight download)
- `KiLocNet` takes `pretrained: bool` — pass `False` in debug scripts
- Only P2 (stride 4) is consumed by heads; P3–P5 returned by FPN but discarded in `kiloc_net.py`
- Simplified to localization only: original draft had 4 heads (loc + count × 2); final has 2 (pos/neg localization only)
- `HeatmapHead` has no sigmoid — outputs raw logits. Loss function must handle this (BCEWithLogitsLoss or apply sigmoid before MSE)

---

## Validation results

- [x] `out.shape == (2, 2, 160, 160)` on random `(2, 3, 640, 640)` input
- [x] No NaN in output
- [x] No Inf in output
- [x] Forward pass on real DataLoader batch: `(4, 2, 160, 160)` ✓
- [x] `loss.backward()` runs without error

---

## Issues encountered

- **Module-level instantiation bug**: original draft had `ResNet34Backbone = build_resnet34_backbone(pretrained=True)` at module level, then `self.backbone = ResNet34Backbone(pretrained=True)` — called forward() on an instance instead of creating a new one. Fixed by calling the factory function directly in `__init__`.
- **Backbone output is a dict**: `create_feature_extractor` returns `{"c2": ..., "c3": ..., ...}`, not a tuple. Original draft unpacked as tuple — fixed.
- **Missing `import torch.nn.functional as F`** in original kiloc_net.py draft — fixed.

---

## Follow-up tasks

- Loss function: decide MSE vs focal loss; implement in `src/kiloc/losses/`
- Training loop: `scripts/train.py` + `src/kiloc/training/`
- Sigmoid/activation: decide whether to add sigmoid inside `HeatmapHead.forward` or handle in loss

---

## Date closed
2026-03-10
