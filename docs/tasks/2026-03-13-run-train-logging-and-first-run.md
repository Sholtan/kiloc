# Run-train logging, config file, and first training run

## Date opened
2026-03-13

## Status
done

---

## Why this task exists

`scripts/run_train.py` was functional but had all hyperparameters hardcoded in the script, no experiment logging, and no way to reproduce a run from its saved artifacts. Before executing real training experiments, the entry point needed to be made reproducible and self-documenting.

Additionally, the first actual training run with `sigmoid_weighted_mse_loss` had not yet been executed.

---

## Task definition

1. Refactor `scripts/run_train.py` to read all hyperparameters from a YAML config file passed via `--config` CLI argument.
2. At the start of each run, create a timestamped run directory under `checkpoint_dir`, copy the config into it, and save per-epoch history to `history.json` (overwritten each epoch).
3. Save model weights on best val F1 (not best val loss).
4. Write `scripts/plot_history.py` to visualize train/val loss and P/R/F1 from a run directory.
5. Execute the first training run with `sigmoid_weighted_mse_loss` and record results.

---

## Desired end state

- `python scripts/run_train.py --config configs/train_1.yaml` runs a full training experiment
- Each run produces a self-contained directory: `config.yaml`, `history.json`, `kilocnet_best_f1_epoch_N.pth`
- `python scripts/plot_history.py --run_dir <path>` saves `losses.png` and `metrics.png`
- First baseline result recorded in `docs/experiments.md`

---

## Relevant files / modules

- `scripts/run_train.py` — training entry point (refactored)
- `scripts/plot_history.py` — new visualization script
- `configs/train_1.yaml` — new config file for first experiment
- `docs/experiments.md` — experiment record

---

## Interfaces / contracts affected

### run_train.py CLI
- Input: `--config <path>` (default: `configs/train_1.yaml`)
- Per-run output directory: `checkpoint_dir/run_YYYYMMDD_HHMMSS/`
  - `config.yaml` — copy of the config used
  - `history.json` — list of per-epoch dicts: `{epoch, train_loss, val_loss, precision, recall, f1}`
  - `kilocnet_best_f1_epoch_N.pth` — weights at best val F1 epoch

### configs/train_1.yaml keys
`loss`, `epochs`, `batch_size`, `num_workers`, `lr`, `weight_decay`, `is_pretrained`, `sigma`, `out_hw`, `in_hw`, `kernel_size`, `threshold`, `merge_radius`, `matching_radius`

---

## What was implemented

- `argparse` with `--config` argument and default
- `yaml.safe_load` inside `main(config_path)`
- Timestamped `run_dir` created with `pathlib.mkdir(parents=True)`
- `shutil.copy(config_path, run_dir / 'config.yaml')` before training starts
- All hyperparameters extracted from `cfg` dict; nothing hardcoded in script body
- Loss function selected via `if/elif` on `cfg['loss']` string; `ValueError` on unknown value
- Checkpoint condition changed from `best_val_loss` to `best_f1`; `best_epoch` tracked
- After loop: checkpoint renamed to `kilocnet_best_f1_epoch_{best_epoch}.pth` via `pathlib.rename`
- `history.json` written (full overwrite) after every epoch
- `scripts/plot_history.py`: loads `history.json`, saves `losses.png` and `metrics.png` to run dir

---

## Issues encountered

### `yaml.safe_load` received path string instead of file object
- **Observation:** `cfg = yaml.safe_load(config_path)` — passed string instead of file handle; reads the path string character by character, returns garbage
- **Fix:** `with open(config_path) as f: cfg = yaml.safe_load(f)`

### `lr: 3e-4` parsed as string by PyYAML
- **Observation:** YAML parses `3e-4` as string in some versions — no decimal point before `e`
- **Fix:** Changed to `3.0e-4` and `1.0e-2` in `train_1.yaml`; added `float()` cast in script as safety net

### `torch.save` target was directory, not file
- **Observation:** `torch.save(model.state_dict(), run_dir)` — passed directory path; crashes
- **Fix:** `torch.save(model.state_dict(), run_dir / "kilocnet_epoch_best.pth")`

### `plt.show()` does nothing on headless server
- **Observation:** `plot_history.py` produced no output on remote machine
- **Fix:** replaced with `fig.savefig(run_dir / "losses.png", dpi=150, bbox_inches='tight')`

### `figures/` subdirectory did not exist
- **Observation:** `fig.savefig(run_dir / "figures/losses.png")` crashed with `FileNotFoundError`
- **Fix:** saved directly to `run_dir` without subdirectory

### Checkpoint saved on best val loss — wrong signal
- **Observation:** val loss rises from epoch ~5 while F1 continues improving — loss/metric decoupling due to model producing sharper peaks than smooth Gaussian targets
- **Fix:** changed checkpoint condition to `f1 > best_f1`

---

## Validation

- [x] `run_train.py` runs end-to-end with `configs/train_1.yaml`
- [x] `run_dir/` contains `config.yaml`, `history.json`, and renamed best checkpoint after run
- [x] `plot_history.py` produces `losses.png` and `metrics.png` from a completed run dir
- [x] First training run completed: 24 epochs, converged at ~epoch 15

---

## What was learned

- Val loss and F1 decouple with weighted MSE on heatmaps: model learns to produce sharper peaks than the smooth Gaussian targets, increasing MSE while improving localization. Val loss is not a reliable checkpoint signal for this task.
- Checkpoint condition should be best val F1, not best val loss, for localization heatmap training.
- YAML scalar notation `3e-4` is ambiguous — use `3.0e-4` with explicit decimal for reliable float parsing.
- PyYAML parses `[160, 160]` inline list natively as Python list of ints — no conversion needed.

---

## Follow-up tasks

1. **Second training run** — repeat with `sigmoid_focal_loss` using `configs/train_2.yaml`; compare val F1 curves
2. **Add `CosineAnnealingLR` scheduler** — after focal loss comparison confirms which loss to use
3. **Remove `print_info` debug calls** from `val_one_epoch` (lines 75–76 of `train.py`)
4. **Smoke test `compute_metrics`** — `scripts/debug/debug_metrics.py` exists but needs the `import torch` removed and a success print added
5. **Augmentation** — no augmentation used in baseline; add as next ablation after loss comparison

---

## Date closed
2026-03-13
