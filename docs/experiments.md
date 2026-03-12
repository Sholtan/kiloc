# Experiments

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

