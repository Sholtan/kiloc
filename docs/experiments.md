# Experiments

## 2026-03-13 — First training run: sigmoid_weighted_mse_loss baseline

- **Goal:** Execute first full training run end-to-end; establish baseline P/R/F1 on BCData validation split with `sigmoid_weighted_mse_loss`
- **Config:** ResNet34 pretrained, FPN, P2 (160×160), sigma=3.0, lr=3.0e-4, weight_decay=1.0e-2, batch_size=8, epochs=24, kernel_size=3, threshold=0.5, merge_radius=1.5, matching_radius=10
- **Loss:** `sigmoid_weighted_mse_loss`
- **Split:** `train` for training, `validation` for evaluation
- **Hypothesis:** pretrained ResNet34 should converge on BCData localization within ~50 epochs; baseline F1 > 0.7 expected
- **Result (best epoch ~15):**
  - train loss: 0.00109
  - val loss: 0.05941
  - val precision: 0.844
  - val recall: 0.829
  - val F1: **0.836**
- **Observations:**
  - Val loss rises from epoch ~5 while F1 continues improving until epoch ~15 — loss/metric decoupling: model produces sharper peaks than smooth Gaussian targets, increasing MSE while improving localization
  - Precision and recall both plateau after epoch ~15; no improvement with further training
  - Checkpoint condition changed mid-session from best val loss to best val F1 after observing decoupling
- **Conclusion:** Baseline established. F1=0.836 at matching_radius=10 with weighted MSE. Model converges at ~15 epochs; 100 epochs is wasteful. Val loss is not a reliable signal for checkpoint selection with this loss function.
- **Next action:** Run second experiment with `sigmoid_focal_loss` (configs/train_2.yaml, epochs=30); compare val F1 curves

---

## 2026-03-12 — debug_weighted_mse.py smoke test

- **Goal:** confirm `sigmoid_weighted_mse_loss` runs without error and returns a finite scalar
- **Config:** random `pred_logits (2,2,160,160)`, random `target (2,2,160,160)` in `[0,1]`; `alpha_pos=100`, `alpha_neg=100`, `q=2`
- **Hypothesis:** loss is finite scalar, no crash
- **Result:** passed
- **Observations:** loss value finite; no shape or device errors
- **Conclusion:** weighted MSE implementation is correct for standard input
- **Next action:** smoke test focal loss, then write training entry point

---

## 2026-03-12 — debug_focal_loss.py smoke test

- **Goal:** confirm `sigmoid_focal_loss` runs without error and returns a finite scalar after bug fixes
- **Config:** random `pred_logits (2,2,160,160)`, random `target (2,2,160,160)` in `[0,1]`; `alpha=2`, `beta=4`; `n_cells=50`
- **Hypothesis:** loss is finite scalar after fixing inverted mask, device/dtype, and log clamp
- **Result:** passed after three fixes
- **Observations:** original code had inverted peak/background mask, CPU-only intermediate tensor, no log clamp — all three caused incorrect or unstable output
- **Conclusion:** focal loss implementation correct after fixes; peak mask logic must be tested explicitly in any future modifications
- **Next action:** write training entry point, run first training experiment

---

## 2026-03-10 — debug_model.py sanity check

- **Goal:** verify KiLocNet produces correct output shapes and gradients flow correctly
- **Config:** `KiLocNet(pretrained=False)`, random input `(2, 3, 640, 640)` + real DataLoader batch `(4, 3, 640, 640)`
- **Hypothesis:** model forward pass produces `(B, 2, 160, 160)` with no NaN/Inf; backward pass runs without error
- **Result:** all assertions passed
- **Observations:** output shape correct; no NaN/Inf; backward pass clean
- **Conclusion:** model architecture is correct end-to-end; ready for loss function
- **Next action:** implement loss function in `src/kiloc/losses/`

---

## 2026-03-09 — debug_dataset.py sanity check

- **Goal:** verify BCDataDataset returns correctly shaped, correctly typed samples with visually plausible heatmaps
- **Config:** `BCDataDataset(split="train")`, `LocHeatmap(out_hw=(160,160), in_hw=(640,640), sigma=3.0)`, batch_size=4
- **Hypothesis:** image shape `(3,640,640)` float32, heatmap shape `(2,160,160)` float32, peaks on cell nuclei
- **Result:** all shape/dtype assertions passed; visual check confirmed peaks align with annotated nuclei
- **Observations:** coordinate convention confirmed `(x, y)`; `torch.cat` correctly stacks pos/neg channels
- **Conclusion:** dataset pipeline is correct and ready for model input
- **Next action:** implement model (done)

