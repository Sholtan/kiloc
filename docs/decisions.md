# Decisions

## 08-03-2026 - Use resnet34 as backbone
- Decision: use ResNet34 as the CNN backbone
- Reason: used in a prior project, allows direct comparison
- Alternatives considered: other ResNet variants, UNet, EfficientNet
- Consequence: backbone choice is fixed for the baseline; may be ablated later

---

## 09-03-2026 - Package structure: src/kiloc/ with submodules
- Decision: source code lives under `src/kiloc/` with `datasets/` and `target_generation/` as submodules; installed via `pip install -e .` with `pyproject.toml` `where = ["src"]`
- Reason: standard src-layout; makes `import kiloc` work across scripts and notebooks without path hacks
- Alternatives considered: flat `src/` with no `kiloc/` subdirectory (tried first, failed — setuptools found no package)
- Consequence: all new modules must be placed inside `src/kiloc/`

---

## 09-03-2026 - Heatmap generation via target_transform functor, not in training loop
- Decision: `BCDataDataset` accepts a `target_transform` callable (e.g. `GaussianHeatmapGenerator`) and calls it per sample inside `__getitem__`. Heatmap generation is not done in the training loop.
- Reason: keeps training loop clean; allows swapping heatmap strategy without touching dataset or loop; consistent with torchvision transform pattern; `GaussianHeatmapGenerator` can be tested independently
- Alternatives considered: generating heatmaps in the training loop (rejected: messy); hardcoding inside `__getitem__` (rejected: not swappable)
- Consequence: `GaussianHeatmapGenerator` must be a callable class initialized with `output_size`, `output_stride`, and `sigma`

---

## 09-03-2026 - __getitem__ return contract: (img_tensor, heatmap_tensor)
- Decision: `__getitem__` returns a tuple `(img_tensor, heatmap_tensor)` where `img_tensor` is `(3, 640, 640)` float32 and `heatmap_tensor` is `(2, 160, 160)` float32 (channel 0 = positive, channel 1 = negative)
- Reason: explicit, minimal, DataLoader-compatible
- Alternatives considered: returning a dict (deferred — no need yet)
- Consequence: training loop must unpack accordingly; heatmap channel order must stay consistent across dataset, loss, and decoder

---

## 09-03-2026 - Debug and sanity checks via scripts, not notebooks
- Decision: dataset validation done in `scripts/debug_dataset.py`, not a Jupyter notebook
- Reason: scripts are rerunnable without kernel restarts, version-controlled, and consistent with project entry-point conventions
- Alternatives considered: notebook with `%autoreload` (valid but less clean for repeatable checks)
- Consequence: `scripts/` is the location for all debug/sanity entry points

---

## 10-03-2026 - build_resnet34_backbone as factory function
- Decision: backbone instantiation wrapped in `build_resnet34_backbone(pretrained: bool)` factory function; not a module-level instance
- Reason: module-level instantiation downloads weights at import time; also causes the bug of calling `.forward()` instead of constructing a new module
- Alternatives considered: module-level variable (tried first, caused two bugs)
- Consequence: always call `build_resnet34_backbone(pretrained=...)` to get a backbone instance

---

## 10-03-2026 - KiLocNet: localization only, two heads
- Decision: `KiLocNet` has two heads (`pos_head`, `neg_head`) producing `(B, 2, 160, 160)`; no density or count heads
- Reason: original draft included 4 heads (loc + density × 2); simplified to match stated goal of localization only
- Alternatives considered: 4-head design with density maps and scalar counts (rejected — scope creep for baseline)
- Consequence: `forward()` returns a single tensor `loc_hm (B, 2, 160, 160)`

---

## 10-03-2026 - HeatmapHead outputs raw logits, no final activation
- Decision: `HeatmapHead.forward` ends with a 1×1 conv, no sigmoid
- Reason: defers activation to the loss; allows numerically stable `BCEWithLogitsLoss`
- Alternatives considered: sigmoid inside head (couples head to a specific loss)
- Consequence: loss must use `BCEWithLogitsLoss` or apply `torch.sigmoid` before MSE — must decide before writing loss

---

## 10-03-2026 - configs/paths.yaml gitignored
- Decision: `configs/paths.yaml` is not committed; each machine maintains its own copy
- Reason: contains absolute machine-specific paths
- Alternatives considered: committing with placeholder values; environment variables
- Consequence: new machines must create their own `configs/paths.yaml`

---

## 12-03-2026 - Implement both MSE and focal loss variants; compare by experiment
- Decision: both `sigmoid_weighted_mse_loss` and `sigmoid_focal_loss` are implemented; choice between them deferred to experimental comparison
- Reason: no strong prior for which performs better on BCData at this scale; both are standard choices
- Alternatives considered: MSE only (simpler), focal only (theoretically stronger for sparse targets)
- Consequence: first training experiment must run both and compare val P/R/F1 before committing to one

---

## 12-03-2026 - BCDataDataset.__getitem__ returns 4-tuple with raw point arrays
- Decision: `__getitem__` returns `(img_tensor, heatmap_tensor, pos_pts, neg_pts)` where `pos_pts`/`neg_pts` are `NDArray[np.int64]`
- Reason: `number_of_cells` for focal loss normalization must come from raw annotations (heatmap-derived count is unreliable with large sigma); coordinates also needed for future P/R/F1 evaluation
- Alternatives considered: returning `n_cells` as a scalar (rejected — loses coordinate information needed for evaluation)
- Consequence: requires custom `collate_fn`; training loop unpacks 4-tuple; `debug_dataset.py` needs updating

---

## 12-03-2026 - Focal loss peak mask uses threshold, not equality
- Decision: `mask = target >= 0.99` identifies peak pixels in the focal loss
- Reason: floating point equality `target == 1.0` is fragile; large sigma could produce multiple sub-1.0 peaks
- Alternatives considered: `target == 1.0` (works for current sigma but fragile), `target > 0.5` (too broad)
- Consequence: if sigma is changed significantly, threshold should be revisited

---

## 12-03-2026 - No LR scheduler for baseline
- Decision: fixed learning rate for the first training run; no scheduler
- Reason: adds complexity before confirming the training pipeline works at all
- Alternatives considered: CosineAnnealingLR (preferred long-term), ReduceLROnPlateau
- Consequence: add `CosineAnnealingLR` after first successful training run is confirmed

---

## 12-03-2026 - compute_metrics returns raw counts, not ratios
- Decision: `compute_metrics(pred_pts, gt_pts, radius)` returns `(tp, fp, fn)` as ints, not precision/recall/F1
- Reason: per-sample ratios cannot be correctly averaged across a dataset; counts must be accumulated first, then divided once at epoch level
- Alternatives considered: returning per-sample P/R/F1 and averaging (incorrect for sparse or empty samples)
- Consequence: caller (`val_one_epoch`) is responsible for epoch-level accumulation and final ratio computation

---

## 12-03-2026 - Checkpoint saved on best val loss, not best F1
- Decision: `run_train.py` saves checkpoint when `total_loss_val < best_val_loss`
- Reason: F1 requires a threshold and decoder settings to be meaningful; val loss is a direct model signal available every epoch; simpler for the baseline
- Alternatives considered: best val F1 (preferred long-term once decoder settings are tuned)
- Consequence: saved checkpoint may not correspond to best F1; revisit after first experiments confirm decoder threshold

---

## 13-03-2026 - Checkpoint saved on best val F1, not best val loss
- Decision: checkpoint condition changed from `total_loss_val < best_val_loss` to `f1 > best_f1`
- Reason: val loss and F1 decouple with weighted MSE on heatmaps — model sharpens peaks relative to smooth Gaussian targets, which increases MSE while improving localization; val loss peaked at epoch 5 while F1 continued rising to epoch 15
- Alternatives considered: best val loss (tried first — saves wrong epoch), combined loss+F1 threshold (unnecessary complexity)
- Consequence: saved checkpoint corresponds to best val F1 epoch; val loss curve can be ignored as checkpoint signal for this loss function

---

## 13-03-2026 - Hyperparameters in YAML config, passed via --config CLI argument
- Decision: all hyperparameters moved to `configs/train_N.yaml`; `run_train.py` accepts `--config` via argparse; config is copied into run directory at start
- Reason: hardcoded parameters in scripts are not reproducible; config file becomes the artifact that documents the exact settings of each run
- Alternatives considered: argparse for every parameter (too verbose), environment variables (non-standard for ML configs)
- Consequence: to start a new experiment, create a new `configs/train_N.yaml`; running without `--config` uses `configs/train_1.yaml` as default

---

## 13-03-2026 - Run directory per experiment with timestamp
- Decision: each training run creates `checkpoint_dir/run_YYYYMMDD_HHMMSS/` containing `config.yaml`, `history.json`, and the best checkpoint
- Reason: single checkpoint file gets overwritten across runs; timestamped directories make all runs independently inspectable and comparable
- Alternatives considered: run ID counter (fragile across machines), flat directory with run prefix (harder to browse)
- Consequence: `checkpoint_dir` accumulates one subdirectory per run; each is self-contained and replayable

---

## 12-03-2026 - matching_radius=10 in run_train.py baseline
- Decision: `matching_radius=10` pixels (image space) used in `val_one_epoch` for the first training run
- Reason: conservative choice for initial run; BCData cells are visible at 640×640 and 10px gives reasonable slack for imprecise early-training predictions
- Alternatives considered: `matching_radius=6` (tighter, standard), `matching_radius=15` (too loose)
- Consequence: reported F1 at radius=10 will be higher than at radius=6; compare both once model converges