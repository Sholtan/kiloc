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