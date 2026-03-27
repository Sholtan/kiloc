# Hard-Enriched 3-Class Benchmark with OOF Embeddings

## Purpose

Implement a new subproject that builds a **hard-enriched 3-class crop-classification dataset** using **out-of-fold (OOF) embeddings**.

This subproject is **not** for detector retraining and **not** for matched-vs-unmatched proposal filtering.
Its purpose is to answer a sharper scientific question:

> If annotated positive/negative cells are replaced with annotated cells that are nearest to the mined ambiguous class in OOF embedding space, is the mined class still separable?

This is intended to test whether prior 3-class separability was partly inflated by using many easy annotated samples.

---

## Project context and why this subproject exists

Earlier stages already established the following:

1. A 3-class crop classifier (`annotated_positive`, `annotated_negative`, `mined_rexclude30`) is feasible on internal held-out data, but the mined class is much harder than the annotated classes.
2. The mined class overlaps especially strongly with `annotated_negative`.
3. Detector-side hard-negative fine-tuning did not improve macro F1.
4. Using the 3-class classifier as a second-stage rejector also did not improve detector macro F1 because it removed too many matched proposals and hurt recall.

Interpretation:
- the mined class contains real structure,
- but it is **not** a clean reject/background class,
- and it overlaps with annotated classes, especially the negative class.

Therefore the next step is **not** more detector suppression.
The next step is to build a **hard-enriched benchmark** using OOF embeddings and test whether class-3 separability survives after replacing easy annotated samples with annotated samples that are close to class 3.

---

## High-level goal

Build a pipeline that:

1. trains an initial 3-class crop classifier in **OOF mode**,
2. extracts **OOF embeddings** and **OOF class probabilities** for all training samples,
3. scores each annotated sample by closeness to the mined class (`class 3`),
4. selects hard annotated positives and negatives using those OOF scores,
5. mixes them with random/diverse anchors,
6. writes a final **hard-enriched dataset CSV**,
7. optionally trains a final classifier on that hard-enriched dataset.

---

## Scope boundaries

### In scope
- image-level fold split for the original train images,
- OOF training/inference for the bootstrap 3-class crop model,
- OOF embedding extraction,
- OOF probability extraction,
- class-3 proximity scoring,
- hard-enriched annotated subset construction,
- final dataset CSV generation,
- optional final classifier training on the refined dataset,
- analysis artifacts for sanity checking.

### Out of scope
- detector retraining,
- matched-vs-unmatched proposal filtering,
- proposal-level filter calibration,
- modifying localization code except for shared utilities if truly needed,
- using validation/test images to construct the hard-enriched train set,
- using in-sample embeddings for hard-sample selection.

---

## Terminology

- **class 0**: `annotated_positive`
- **class 1**: `annotated_negative`
- **class 2 / class 3 in discussions**: `mined_rexclude30` (mined ambiguous / unannotated hard detections)

Important: class 2 is **not** “true background” and **not** “confirmed non-tumor.”
It is a mined ambiguous class.

---

## Data sources

Use only the **original training images** for the OOF hard-enrichment pipeline.

Required source pools:

1. `annotated_positive_pool`
2. `annotated_negative_pool`
3. `mined_class3_pool`

Each row must include at least:
- `image_id`
- `point_id` (or deterministic unique row key)
- `x`
- `y`
- `class_label`
- source information

For mined samples also keep:
- detector score(s)
- detector predicted class
- mining metadata if available

Do **not** use internal validation/test or official validation/test images to build the training hard-enriched set.

---

## Mandatory split policy

Everything must be split by **image ID**, never by crop.

Create one fixed fold assignment table for the original train images:

- `image_id`
- `fold`

Recommended default: `K = 5`.

This same fold assignment must be used consistently for:
- bootstrap OOF classifier training,
- OOF embedding extraction,
- OOF probability extraction,
- all downstream selection scores.

---

## Folder / file layout (proposed)

Create the new code under the context-radius / three-class area and keep it isolated from localization code.

Proposed structure:

```text
src/three_classifier/
    datasets/
        hard_enriched_dataset.py
    models/
        crop_encoder.py
    utils/
        embedding_io.py
        nearest_neighbors.py
        selection_scores.py
        hard_enriched_sampling.py

scripts/three_classifier/
    run_build_oof_folds.py
    run_train_bootstrap_oof_classifier.py
    run_extract_oof_embeddings.py
    run_build_hard_enriched_set.py
    run_train_hard_enriched_classifier.py
    run_export_hard_examples.py

configs/three_classifier/
    hard_enriched_oof_bootstrap.yaml
    hard_enriched_oof_final.yaml
```

Reuse existing dataset/model/training utilities when appropriate, but do not break the previous subproject.

---

## Phase A: bootstrap OOF classifier

### Objective
Train a first-pass 3-class crop classifier only to obtain **OOF embeddings** and **OOF class probabilities**.

### Bootstrap dataset definition
Use:
- class 0: random annotated positives,
- class 1: random annotated negatives,
- class 2: mined class-3 samples.

Balanced influence across classes is required.
Implementation can use either:
- balanced downsampling once, or
- balanced batch sampling.

For the bootstrap stage, random annotated sampling is acceptable.

### Model requirements
- backbone: configurable, default `resnet18`
- embedding dimension: `128`
- classifier head: linear `128 -> 3`
- output both:
  - logits
  - L2-normalized embedding

### Training requirements
- image-level fold-safe training
- one model per fold
- train on folds `!= f`, infer on fold `f`
- save best checkpoint per fold

### Required outputs per held-out sample
Save one row per sample with:
- `image_id`
- `point_id`
- `x`
- `y`
- `true_class`
- `fold`
- `prob_pos`
- `prob_neg`
- `prob_class3`
- `pred_class`
- `embedding_0 ... embedding_127`

Write a concatenated OOF artifact, e.g.:
- `oof_bootstrap_predictions.csv`
- `oof_bootstrap_embeddings.npy` or one CSV with embeddings inline

---

## Phase B: OOF proximity scoring to class 3

### Objective
Assign each annotated sample a score indicating how close it is to the mined class manifold.

### Embedding normalization
All embeddings must be L2-normalized before nearest-neighbor search.

### Nearest-neighbor index
Build a nearest-neighbor index over **class-3 embeddings only**.

Cosine similarity is preferred.

### Per-annotated-sample scores
For each annotated sample `a`, compute at least:

1. `top1_class3_cosine`
2. `mean_top5_class3_cosine`
3. `prob_class3`
4. `pred_class`

Optional but useful:
5. entropy
6. top-1 / top-2 margin
7. IDs of nearest class-3 neighbors

### Recommended combined hard score
Use a weighted combination such as:

```text
hard_score = 0.7 * mean_top5_class3_cosine + 0.3 * prob_class3
```

Make weights configurable.

Keep the implementation modular so alternative score definitions can be swapped in later.

---

## Phase C: hard-enriched subset construction

### Objective
Construct new annotated positive/negative subsets enriched with samples close to class 3.

### Selection policy
Process positive and negative annotated classes **separately**.

For each annotated class:
1. rank samples by `hard_score` descending,
2. apply diversity constraints,
3. select a hard subset,
4. add random/diverse anchors.

### Default recommended mixing ratio
Configurable, with default:
- `hard_fraction = 0.6`
- `anchor_fraction = 0.4`

Allow easy switching to `0.5 / 0.5`.

### Diversity constraints
Must implement at least:
- per-image cap (`max_selected_per_image`)

Optional but preferred:
- embedding-space diversity cap or clustering,
- spatial deduplication within image.

### Mined class handling
Use all mined class-3 samples by default, unless a cap is explicitly configured.

### Final dataset composition
- class 0 = hard-enriched positives + positive anchors
- class 1 = hard-enriched negatives + negative anchors
- class 2 = mined class-3 samples

---

## Final dataset CSV schema

Write one definitive CSV that fully specifies the final dataset.

Required columns:
- `image_id`
- `point_id`
- `x`
- `y`
- `final_class`
- `source_pool` (`annotated_positive`, `annotated_negative`, `mined_class3`)
- `selection_type` (`hard_selected`, `anchor_random`, `mined_original`)
- `fold_origin`
- `prob_pos_oof`
- `prob_neg_oof`
- `prob_class3_oof`
- `pred_class_oof`
- `top1_class3_cosine`
- `mean_top5_class3_cosine`
- `hard_score`

Optional columns:
- detector score fields for class 3,
- nearest class-3 neighbor IDs,
- entropy,
- margin.

Suggested file name:
- `hard_enriched_train_set.csv`

---

## Phase D: final classifier training (optional but should be supported)

### Objective
Train a new classifier **from scratch** on the hard-enriched dataset.

Important:
- do **not** continue from the bootstrap fold model,
- bootstrap is for selection only,
- final model is for evaluation on the refined task.

### Required outputs
- final checkpoint(s)
- validation/test metrics
- confusion matrix
- prediction CSV with probabilities
- exported hardest samples for review

---

## Evaluation requirements

Support two evaluation modes:

### 1. Natural held-out evaluation
Evaluate on a regular held-out split to check that the model is still meaningful beyond the curated hard benchmark.

### 2. Hard-enriched evaluation
Evaluate on a corresponding hard-enriched held-out benchmark if configured.

At minimum report:
- macro F1
- per-class precision/recall/F1
- confusion matrix
- support per class

---

## Analysis exports

Implement exports to support manual review and paper figures.

### Required review buckets
Export crops / rows for at least:
- annotated positive with highest `prob_class3`
- annotated negative with highest `prob_class3`
- annotated positive with highest `mean_top5_class3_cosine`
- annotated negative with highest `mean_top5_class3_cosine`
- random hard-selected positives
- random hard-selected negatives
- random class-3 samples

### Purpose
These review sets are needed to check whether the selected boundary-near annotated subset contains:
- genuinely ambiguous tumor cells,
- likely missannotations,
- negative-like nuclei,
- artifacts.

---

## Configuration parameters that must be exposed

At minimum:
- `folds`
- `backbone`
- `embedding_dim`
- `crop_size`
- `resize_h`
- `resize_w`
- `batch_size`
- `epochs`
- `optimizer`
- `lr`
- `weight_decay`
- `scheduler`
- `class_balance_mode`
- `hard_fraction`
- `anchor_fraction`
- `max_selected_per_image`
- `knn_k`
- `hard_score_weight_cosine`
- `hard_score_weight_prob_class3`
- `use_all_mined_samples`
- paths to input CSVs / splits / outputs

---

## Implementation constraints

1. Do not modify localization training code unless absolutely necessary for shared utilities.
2. Keep all new logic isolated in the crop-classification subproject.
3. No leakage across image splits.
4. No in-sample embeddings for hard-sample selection.
5. The pipeline must be runnable end-to-end from config/CLI.
6. Outputs must be deterministic given seed.

---

## Acceptance criteria

The subproject is complete when all of the following are true:

1. A fixed image-level fold file can be built for the original train images.
2. A bootstrap 3-class classifier can be trained in OOF mode.
3. OOF embeddings and OOF probabilities are written for every training sample.
4. Annotated samples receive class-3 proximity scores.
5. A hard-enriched dataset CSV is written with the required schema.
6. A final classifier can be trained from scratch on the hard-enriched set.
7. Metrics and confusion matrices are exported.
8. Review buckets can be exported for manual inspection.
9. The pipeline can be run from CLI without hand-editing code.

---

## Recommended CLI sequence

Example target workflow:

```bash
python scripts/three_classifier/run_build_oof_folds.py \
  --config configs/three_classifier/hard_enriched_oof_bootstrap.yaml

python scripts/three_classifier/run_train_bootstrap_oof_classifier.py \
  --config configs/three_classifier/hard_enriched_oof_bootstrap.yaml

python scripts/three_classifier/run_extract_oof_embeddings.py \
  --config configs/three_classifier/hard_enriched_oof_bootstrap.yaml

python scripts/three_classifier/run_build_hard_enriched_set.py \
  --config configs/three_classifier/hard_enriched_oof_final.yaml

python scripts/three_classifier/run_train_hard_enriched_classifier.py \
  --config configs/three_classifier/hard_enriched_oof_final.yaml
```

---

## Primary scientific question this subproject should answer

This subproject is successful if it lets us answer the following:

> When annotated positives/negatives are replaced by annotated samples nearest to the mined ambiguous class in OOF embedding space, does class 3 remain separable, or does the apparent separation collapse?

Interpretation guide:

- If class 3 remains separable, then the mined class has structured differences beyond easy-vs-hard imbalance.
- If separability drops sharply, then prior 3-class performance was partly driven by easy annotated samples.
- If overlap is strongest with the negative class, that supports the current project finding that the ambiguous region is especially close to annotated negatives.

---

## Notes for Codex

- Reuse existing repo utilities and style conventions when possible.
- Prefer small, composable modules over one monolithic script.
- Write clear CSV schemas and save enough metadata for downstream analysis.
- Do not silently change previous experiment logic.
- Keep the bootstrap stage and final hard-enriched stage clearly separated.
- The OOF requirement is the most important correctness constraint in this subproject.

