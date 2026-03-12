# Loss functions and training loop

## Date opened
2026-03-12

## Status
done

---

## Why this task exists

With the dataset and model verified, the next step in the baseline pipeline is the loss function and the training loop. Both are required before any training experiment can be run.

---

## Task definition

Implement two heatmap localization loss functions (`sigmoid_weighted_mse_loss`, `sigmoid_focal_loss`) in `src/kiloc/losses/losses.py`. Implement `train_one_epoch` in `src/kiloc/training/train.py`. Update `BCDataDataset.__getitem__` to return point coordinate arrays alongside heatmaps for use in the loss and future evaluation.

---

## Desired end state

- Both loss functions pass smoke tests with correct input shapes and dtypes
- `train_one_epoch` runs a full epoch, returns average loss, applies gradient clipping
- `BCDataDataset.__getitem__` returns `(img_tensor, heatmap_tensor, pos_pts, neg_pts)`
- Custom `collate_fn` handles variable-length point arrays in DataLoader
- `val_one_epoch` deferred — requires evaluation module first

---

## Out of scope

- Validation loop (requires `evaluation/decode.py` and `evaluation/metrics.py`)
- `scripts/train.py` entry point
- LR scheduler
- First training run

---

## Assumptions

- `KiLocNet.forward()` returns raw logits `(B, 2, 160, 160)` — no sigmoid inside the model
- Both losses apply sigmoid internally; they do not expect probabilities as input
- Heatmap channel 0 = positive cells, channel 1 = negative cells
- Coordinate convention: `(x, y)` in annotation files
- `number_of_cells` for focal loss normalization is computed from raw point arrays, not from heatmap threshold (threshold-based approach is unreliable when sigma is large)

---

## Relevant files / modules

- `src/kiloc/losses/losses.py` — both loss functions
- `src/kiloc/training/train.py` — `train_one_epoch`
- `src/kiloc/datasets/bcdata.py` — updated `__getitem__` and `collate_fn`
- `scripts/debug_weighted_mse.py` — smoke test for weighted MSE
- `scripts/debug_focal_loss.py` — smoke test for focal loss

---

## Interfaces / contracts

### sigmoid_weighted_mse_loss
- Signature: `(pred_logits, target, n_cells=None, alpha_pos=100., alpha_neg=100., q=2.) -> Tensor`
- `pred_logits`: `(B, 2, H, W)` raw logits
- `target`: `(B, 2, H, W)` float32 gaussian heatmaps in `[0, 1]`
- `n_cells`: unused; present for interface compatibility with focal loss
- Returns: scalar loss tensor

### sigmoid_focal_loss
- Signature: `(pred_logits, target, n_cells, alpha=2., beta=4.) -> Tensor`
- `pred_logits`: `(B, 2, H, W)` raw logits
- `target`: `(B, 2, H, W)` float32 gaussian heatmaps in `[0, 1]`
- `n_cells`: int — total number of annotated cells in the batch, used for normalization
- Returns: scalar loss tensor
- Peak mask: `target >= 0.99` (threshold, not `== 1.0`)

### BCDataDataset.__getitem__ (updated)
- Output: `(img_tensor, heatmap_tensor, pos_pts, neg_pts)`
  - `img_tensor`: `(3, 640, 640)` float32
  - `heatmap_tensor`: `(2, 160, 160)` float32
  - `pos_pts`: `NDArray[np.int64]` shape `(N_pos, 2)`, convention `(x, y)`
  - `neg_pts`: `NDArray[np.int64]` shape `(N_neg, 2)`, convention `(x, y)`
- `collate_fn` stacks tensors, keeps pts as lists of arrays

### train_one_epoch
- Signature: `(model, criterion, optimizer, device, trainloader) -> float`
- Returns average loss over the epoch
- Applies `clip_grad_norm_(max_norm=1.0)` between `backward()` and `step()`
- Uses tqdm progress bar

---

## What was implemented

- `sigmoid_weighted_mse_loss`: spatial weighting via `w = 1 + alpha * target^q`; per-channel alpha; normalized weighted average loss
- `sigmoid_focal_loss`: CornerNet/CenterNet formulation; peak mask `target >= 0.99`; normalized by `n_cells`; log-clamped for numerical stability; `torch.zeros` with explicit device/dtype
- `train_one_epoch`: full epoch loop, loss accumulation, gradient clipping, tqdm, returns average loss
- `BCDataDataset.__getitem__`: extended to 4-tuple
- `collate_fn`: handles variable-length point arrays

---

## Issues encountered

### Focal loss mask inverted
- **Observation:** `mask = target != 1.` used peak term for background and background term for peaks
- **Cause:** boolean logic error
- **Fix:** `mask = target >= 0.99`; swap which term applies to which mask

### `torch.empty` without device/dtype
- **Observation:** intermediate loss tensor always created on CPU
- **Cause:** `torch.empty(shape)` defaults to CPU float32
- **Fix:** `torch.zeros(shape, dtype=pred.dtype, device=pred.device)`

### Log instability
- **Observation:** `log(pred)` could produce `-inf` when `pred ≈ 0`
- **Fix:** `pred = pred.clamp(min=1e-6, max=1-1e-6)` before log

### `number_of_cells` from heatmap threshold unreliable
- **Observation:** with large sigma, multiple pixels exceed any threshold per cell, giving count > actual cells
- **Fix:** compute from raw point arrays: `sum(len(p) for p in pos_pts) + sum(len(n) for n in neg_pts)`

### `total_loss` not returned from `train_one_epoch`
- **Observation:** computed but function returned `None`
- **Fix:** added `return total_loss`

---

## Validation

- [x] `debug_weighted_mse.py` runs without error; returns finite scalar loss
- [x] `debug_focal_loss.py` runs without error after three bug fixes; returns finite scalar loss
- [ ] full training run not yet performed

---

## What was learned

- Focal loss mask logic must be verified explicitly: peak pixels (`target >= 0.99`) get the penalizing term `-(1-p)^α log p`; near-peak background pixels get the down-weighted term `-(1-target)^β p^α log(1-p)`
- `torch.empty` is unsafe for intermediate tensors that will be partially filled — use `torch.zeros`
- Number-of-cells normalization must come from annotations, not derived from the heatmap itself, because sigma controls how many pixels exceed any threshold

---

## Follow-up tasks

1. **Evaluation module** — `src/kiloc/evaluation/decode.py` (heatmap → predicted points via local maxima) and `src/kiloc/evaluation/metrics.py` (P/R/F1 with distance threshold matching)
2. **`val_one_epoch`** — in `src/kiloc/training/train.py`; requires evaluation module
3. **`scripts/train.py`** — entry point: instantiate dataset, model, optimizer, loss (via `partial`), call train/val loops
4. **First training experiment** — run with both loss variants; log train/val loss curves; compare

---

## Date closed
2026-03-12
