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




## Ideas / backlog
- Need to write separate metrics for negative and positive
- the alpha_pos and alpha_neg need to be adjustable from the config. Adjust them according to the metrics for pos and neg
- add saving of plot of single sample for every epoch
- add visualization of overlay of the best model at the end
- compute Ki-67 index from GT and predicted cell count
- add metrics for Ki-67 index

Answer question: how can we align the loss with the our targeted metric?
Can i run grid search of parameters with my model, how to set it up (single run doesn't take much time)?

what is the model first focuses on the easy task (positives) and then on the hard task (negatives)?

Will warm up help (what is warm up in general)? Should try with/without.



context_radius_study crops:
64, 96, 128, 192
extra small: 32

fixed classifier input: 128×128

