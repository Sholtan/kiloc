# BCData dataset class and heatmap generation setup

## Date opened
2026-03-09

## Status
done

---

## Why this task exists

Stage 1 of the roadmap requires a working `BCDataset` class that loads images and annotations, and a `GaussianHeatmapGenerator` that converts point annotations into localization heatmaps. These are the foundation for all subsequent training and evaluation work.

---

## Task definition

Implement `BCDataDataset` in `src/kiloc/datasets/bcdata.py` and `GaussianHeatmapGenerator` in `src/kiloc/target_generation/heatmaps.py`. Write a debug script at `scripts/debug_dataset.py` to validate both. The dataset must return `(img_tensor, heatmap_tensor)` per sample, ready for the DataLoader.

---

## Desired end state

- `BCDataDataset` loads images and point annotations correctly for all three splits
- `GaussianHeatmapGenerator` produces correct `(160, 160)` float32 heatmaps from point coordinates
- `scripts/debug_dataset.py` runs without error and validates shapes, dtypes, and a visual sanity check
- No debug prints remain in production code
- All files committed on `feat/bcdata-dataset`

---

## Out of scope

- Training loop
- Model definition
- Augmentations (joint_transform, image_transform can remain None for now)

---

## Assumptions currently in force

- Image size: `(640, 640)`, 3-channel RGB
- Heatmap size: `(160, 160)` — output stride 4
- BCData annotation files use key `"coordinates"` in each `.h5` file
- Each image has exactly one `.h5` file in `annotations/<split>/positive/` and one in `annotations/<split>/negative/`, matched by filename stem
- Images are `.png` format
- **Coordinate convention: `(x, y)`** — confirmed by visual inspection. Heatmap indexed as `heatmap[y, x]`.

---

## Relevant files/modules

- `src/kiloc/datasets/bcdata.py` — `BCDataDataset` class
- `src/kiloc/datasets/__init__.py` — package init
- `src/kiloc/target_generation/heatmaps.py` — `LocHeatmap` class
- `src/kiloc/target_generation/__init__.py` — package init
- `src/kiloc/utils/config.py` — `get_paths(device)` returns `(data_root, checkpoint_path)`
- `src/kiloc/utils/debug.py` — `print_info()` helper
- `src/kiloc/visualization/plots.py` — visualization helpers
- `scripts/debug_dataset.py` — debug/sanity check script

---

## Interfaces / contracts affected

### BCDataDataset.__getitem__
- Input: integer index
- Output: `(img_tensor, heatmap_tensor)`
  - `img_tensor`: `torch.Tensor`, shape `(3, 640, 640)`, dtype `float32`, values in `[0, 1]`
  - `heatmap_tensor`: `torch.Tensor`, shape `(2, 160, 160)`, dtype `float32`, values in `[0, 1]`
  - Channel 0 = positive cells, channel 1 = negative cells
- Invariant: both tensors must be contiguous float32

### LocHeatmap.__call__
- Input: `points` — `NDArray[np.int64]`, shape `(N, 2)`, convention `(x, y)`
- Output: `torch.Tensor`, shape `(1, 160, 160)`, dtype `float32`, values in `[0, 1]`
- Initialized with: `out_hw=(160, 160)`, `in_hw=(640, 640)`, `sigma=3.0`, `dtype=torch.float32`
- Uses `torch.maximum` (not sum) to handle overlapping gaussians
- `BCDataDataset.__getitem__` stacks pos/neg with `torch.cat` → `(2, 160, 160)`

---

## What has already been completed

- Package structure established: `src/kiloc/` with `datasets/` submodule
- `pip install -e .` works; `from kiloc.datasets.bcdata import BCDataDataset` imports correctly
- `BCDataDataset.__init__`: split validation, directory path setup — done
- `BCDataDataset._build_index`: image/annotation pairing with existence checks — done
- `BCDataDataset.__len__` — done
- `BCDataDataset._load_points`: reads `"coordinates"` key from `.h5`, returns `NDArray[np.int64]` — done
- `BCDataDataset.__getitem__`: image loading, BGR→RGB, float32 scaling, tensor conversion, point loading, heatmap stacking, return of `(img_tensor, heatmap_tensor)` — structurally complete but blocked on `GaussianHeatmapGenerator`
- Design decision confirmed: heatmap generation stays in `__getitem__` via `target_transform` functor, not in the training loop

## Deferred items

- `target_transform=None` optional path not implemented — `target_transform` is still required. Low priority; defer to when visualization tooling needs it.

---

## Validation results

- [x] `len(dataset)` matches expected number of training images
- [x] `img_tensor.shape == (3, 640, 640)` and `dtype == float32`
- [x] `heatmap_tensor.shape == (2, 160, 160)` and `dtype == float32`
- [x] heatmap values in `[0, 1]`
- [x] gaussian peaks visually align with cell centers — confirmed via `images/debug_sample0.png`
- [x] no crash on first 10 samples
- [x] `DataLoader(dataset, batch_size=4)` produces correctly batched tensors

---

## Risks / failure modes

- Coordinate convention mismatch: heatmap peaks will be transposed or mirrored — catches visually but not by shape checks
- Samples with zero annotations (empty `.h5`) must not crash the generator
- Gaussian overflow: multiple nearby cells can sum above 1.0 — decide whether to clamp or use peak-only (max, not sum)
- `cv2.imread` returns `None` silently for bad paths — already handled with RuntimeError

---

## Debug notes

- Coordinate convention confirmed `(x, y)` via visual inspection of annotation overlaid on image.
- `LocHeatmap` returns `torch.Tensor (1, H, W)` directly; `BCDataDataset.__getitem__` uses `torch.cat` to stack pos/neg channels into `(2, H, W)`.

---

## Definition of done

- [x] `scripts/debug_dataset.py` runs clean with no prints or errors
- [x] shape/dtype assertions pass
- [x] visual sanity check: heatmap peaks align with annotated cell centers
- [x] no debug prints in any production file
- [ ] `target_transform=None` path works without crashing — deferred
- [ ] all files staged and committed on `feat/bcdata-dataset` — pending commit

---

## Date closed
2026-03-10
