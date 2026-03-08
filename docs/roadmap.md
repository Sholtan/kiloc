# in what order you will build it

stage 1: Dataset and sample contract
- implement BCData dataset class
- define sample output format
- heatmap generation
- verify annotation parsing
- verify train/val/test split handling

stage 3: Baseline model
- backbone
- FPN
- loc head
- forward output


stage 4: Training loop
- optimizer, scheduler, logging, checkpointing
- run one-batch overfit test
- run short debug training


stage 5: Evaluation pipeline
- decoder from heatmap to points
- implement matching
- implement precision / recall / F1


stage 6: Baseline stabilization
- debub poor predictions
- inspect targets, outputs, decoding, thresholds
- fix implementation issues


stage 6: Ablations
- sigma
- output resolution
- threshold
- NMS/peak decoding settings
- loss variants
- augmentations