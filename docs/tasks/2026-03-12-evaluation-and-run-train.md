# Evaluation module and training entry point

## Date opened
2026-03-12

## Status
done

---

## Why this task exists

With loss functions and `train_one_epoch` in place, the pipeline needed three remaining pieces before a training run could be executed: a heatmap decoder, a metrics function, `val_one_epoch`, and the `scripts/run_train.py` entry point.

---

## Task definition

Implement `src/kiloc/evaluation/decode.py` (heatmap → predicted point coordinates), `src/kiloc/evaluation/metrics.py` (per-sample P/R/F1 counting), `val_one_epoch` in `src/kiloc/training/train.py`, and `scripts/run_train.py`.

---

## Desired end state

- `heatmaps_to_points_batch` decodes `(B, 2, H, W)` heatmaps to two lists of `(N, 2)` float32 arrays in image space
- `compute_metrics` returns `(tp, fp, fn)` for a single sample via greedy distance matching
- `val_one_epoch` runs a full val epoch, accumulates tp/fp/fn across all samples, returns `(val_loss, precision, recall, f1)`
- `scripts/run_train.py` wires dataset, model, optimizer, criterion, and train/val loops end-to-end; saves checkpoint on best val loss

---

## Out of scope

- First actual training run (requires GPU access, deferred)
- LR scheduler
- Mixed precision

---

## Assumptions

- Decoded coordinates are in image space `(0–639)` — rescaling applied inside `heatmaps_to_points_batch` via `_rescale_points`
- `compute_metrics` returns raw counts, not ratios; epoch-level ratios computed after accumulation
- GT point arrays from `BCDataDataset.__getitem__` are `np.int64` in image space; cast to `float32` before passing to `compute_metrics`
- Checkpoint condition: best val loss (not best F1)

---

## Relevant files / modules

- `src/kiloc/evaluation/decode.py` — heatmap decoder
- `src/kiloc/evaluation/metrics.py` — per-sample tp/fp/fn
- `src/kiloc/training/train.py` — `val_one_epoch`
- `scripts/run_train.py` — training entry point

---

## Interfaces / contracts

### heatmaps_to_points_batch
- Signature: `(heatmaps, kernel_size, threshold, output_hw=(640,640), merge_radius=1.5, refine=True) -> tuple[list, list]`
- `heatmaps`: `(B, 2, H, W)` float32 probabilities (post-sigmoid)
- Returns: `(out_pos, out_neg)` — each a list of `B` arrays, each array `(N, 2)` float32 in `(x, y)` image-space coordinates

### compute_metrics
- Signature: `(pred_pts, gt_pts, radius) -> tuple[int, int, int]`
- Returns `(tp, fp, fn)` for one sample and one channel
- Greedy matching via `scipy.spatial.distance.cdist`; each GT matched at most once

### val_one_epoch
- Signature: `(model, criterion, device, val_loader, kernel_size, threshold, merge_radius, matching_radius) -> tuple[float, float, float, float]`
- Returns `(val_loss, precision, recall, f1)` accumulated over the full epoch

---

## What was implemented

- `decode.py`: `heatmaps_to_points_batch` → `heatmaps_to_points` → `_channel_to_points`; internal helpers `_local_maxima` (max_pool2d), `_merge_close_points` (greedy NMS by score), `_refine_points` (weighted centroid), `_rescale_points`
- `metrics.py`: `compute_metrics` with explicit empty-array handling and greedy `cdist` matching
- `val_one_epoch`: sigmoid applied to logits before decoding; separate pos/neg accumulation; combined P/R/F1 at epoch end
- `run_train.py`: full wiring — paths, dataloaders, model, optimizer, criterion, epoch loop, print logging, best-val-loss checkpoint saving

---

## Issues encountered

### val_one_epoch: criterion call missing heatmaps_batch
- **Observation:** `criterion(logits, pos_pts_tuple, neg_pts_tuple)` — second argument was points tuple, not heatmaps
- **Fix:** `criterion(logits, heatmaps_batch, pos_pts_tuple, neg_pts_tuple)` — matches train_one_epoch signature

### val_one_epoch: matching_radius confused with merge_radius
- **Observation:** `compute_metrics(..., radius=merge_radius)` used 1.5 (heatmap-space NMS radius) instead of 6.0 (image-space matching radius)
- **Fix:** `radius=matching_radius`

### run_train.py: syntax error in checkpoint path
- **Observation:** `checkpoint_dir + / "/checkpoint/kilocnet_v0_epoch_latest"` — invalid Python
- **Fix:** `checkpoint_dir / "kilocnet_v0_best.pth"` using pathlib

### run_train.py: torch.save never called
- **Observation:** checkpoint path computed but save not written
- **Fix:** added `torch.save(model.state_dict(), checkpoint_path)`

### run_train.py: best_val_loss never updated
- **Observation:** condition fires every epoch because `best_val_loss` stayed `np.inf`
- **Fix:** `best_val_loss = total_loss_val` inside the save block

### run_train.py: model.to(device) missing
- **Observation:** model built on CPU, never moved to GPU
- **Fix:** `model = model.to(device)` after device detection

### run_train.py: metrics dict used float values as keys
- **Observation:** `metrics[precision] = precision` — float as dict key
- **Fix:** `metrics["precision"] = precision`

---

## Validation

- [x] `debug_decode.py` — synthetic peak test and real-batch visual inspection passed; decoded point count matched GT count (51 positive cells)
- [ ] `compute_metrics` — no dedicated smoke test written; logic reviewed manually
- [ ] Full training run — not yet executed (GPU access required)

---

## What was learned

- `val_one_epoch` must use `matching_radius` (image-space, ~6–10 px) not `merge_radius` (heatmap-space NMS, ~1.5 px) when calling `compute_metrics`; conflating the two gives near-zero recall
- `compute_metrics` should return raw counts (tp, fp, fn), not ratios, so the caller can accumulate correctly across an epoch before dividing
- GT point arrays from `__getitem__` are `np.int64`; must cast to `float32` before passing to `cdist`

---

## Follow-up tasks

1. **First training run** — `python scripts/run_train.py` with `sigmoid_weighted_mse_loss`; log train/val loss and P/R/F1 per epoch
2. **Second training run** — repeat with `sigmoid_focal_loss`; compare val F1 curves
3. **Smoke test `compute_metrics`** — write `scripts/debug/debug_metrics.py` with toy pred/gt arrays
4. **Remove debug print_info calls** — `print_info` in `val_one_epoch` (lines 75–76 of `train.py`) should be removed before a real run
5. **LR scheduler** — add `CosineAnnealingLR` after first successful training run is confirmed

---

## Date closed
2026-03-12
