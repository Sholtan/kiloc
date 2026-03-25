Assumptions:

1. You want to mine only false positives from the training split, out-of-fold, and use them later to improve the localization model.
2. Your detector outputs class-specific point predictions for Ki-67 positive and Ki-67 negative.
3. You want to avoid contaminating the mined pool with split peaks near a real annotated cell, inter-class conflicts, and obvious decoder artifacts.
4. Matching is point-based with some evaluation radius rmatchr_{\text{match}}rmatch​ such as 20 px.

This is the pipeline I would use.

Use 5-fold image-level OOF mining on the training split only. For each fold, train on 4/5 of training images and infer on the held-out 1/5. Decode points exactly as you do at evaluation. Match predictions to GT one-to-one. Then mine only unmatched predictions that survive a stricter set of filters. After all folds, concatenate the kept candidates into one OOF-mined false-positive table. Then visualize that table before using it for retraining.

The key design choice is this: do not treat every unmatched prediction as a mined negative. A large fraction of those will be near-GT split peaks, cross-class confusion near a real cell, or multiple local maxima on one structure. Those should be filtered or at least marked separately.

I would store three tables, not one.

First table: raw OOF predictions. This is the full decoded output on held-out images.

Suggested CSV schema for `oof_raw_predictions.csv`:

- `fold`
- `image_id`
- `pred_id`
- `pred_class` (`pos` or `neg`)
- `x`
- `y`
- `score`
- `score_pos`
- `score_neg`
- `decode_threshold`
- `model_checkpoint`
- `is_border` (optional)
- `raw_rank_in_image` (optional, by descending score)

Second table: prediction-to-GT relation table. This is the diagnostic table you will use to decide what kind of failure each prediction is.

Suggested CSV schema for `oof_prediction_relations.csv`:

- `fold`
- `image_id`
- `pred_id`
- `pred_class`
- `x`
- `y`
- `score`
- `matched_gt_id` (`""` if unmatched)
- `matched_gt_class` (`""` if unmatched)
- `matched_dist`
- `nearest_gt_any_id`
- `nearest_gt_any_class`
- `nearest_gt_any_dist`
- `nearest_gt_same_id`
- `nearest_gt_same_dist`
- `nearest_gt_other_id`
- `nearest_gt_other_dist`
- `n_preds_sameclass_within_r_cluster`
- `n_preds_anyclass_within_r_interclass`
- `cluster_sameclass_id`
- `cluster_anyclass_id`
- `filter_status`
- `filter_reason`

Third table: final mined false positives only.

Suggested CSV schema for `oof_mined_false_positives.csv`:

- `fold`
- `image_id`
- `pred_id`
- `pred_class`
- `x`
- `y`
- `score`
- `score_pos`
- `score_neg`
- `nearest_gt_any_class`
- `nearest_gt_any_dist`
- `nearest_gt_same_dist`
- `nearest_gt_other_dist`
- `sameclass_cluster_size`
- `anyclass_cluster_size`
- `mining_tag`
- `crop_size_for_review` (optional)
- `review_status` (empty initially; later manual annotation)
- `review_note` (empty initially)

The core algorithm per fold should be this.

Step 1. Split by image, not by point.  
All points from one image must stay in the same fold.

Step 2. Train fold model on 4 folds.

Step 3. Infer on held-out fold and decode predicted peaks per class.

Step 4. Run one-to-one matching against GT with class-aware matching first.  
For example, positive predictions match only positive GT, negative predictions match only negative GT, within radius rmatchr_{\text{match}}rmatch​. Use Hungarian or greedy-by-score; either is fine as long as it is consistent.

Step 5. For every prediction, compute distances to:

- nearest GT of any class,
- nearest GT of same class,
- nearest GT of opposite class.

That lets you separate genuine background-like false positives from “predicted the wrong class on a real cell.”

Step 6. Apply filtering rules to unmatched predictions.

These are the filtering rules I recommend.

Rule A: matched predictions are never mined.  
If a prediction matched a GT, discard from mining.

Rule B: exclude anything too close to any GT point.  
If an unmatched prediction is within

rexclude-any>rmatchr_{\text{exclude-any}} > r_{\text{match}}rexclude-any​>rmatch​

of any GT point, do not mine it as a negative.

This is the most important anti-split rule. It removes the case where two predicted peaks sit around one GT point and one of them would otherwise be counted as a false positive.

A good starting value is

rexclude-any=1.5 rmatch  to  2.0 rmatch.r_{\text{exclude-any}} = 1.5\, r_{\text{match}} \;\text{to}\; 2.0\, r_{\text{match}}.rexclude-any​=1.5rmatch​to2.0rmatch​.

If rmatch=20r_{\text{match}}=20rmatch​=20, start with 30 or 35 px.

Rule C: separate same-class duplicates away from GT.  
Among remaining unmatched predictions of the same class within one image, cluster predictions with radius

rcluster-same≈0.75 rmatch to 1.0 rmatch.r_{\text{cluster-same}} \approx 0.75\,r_{\text{match}} \text{ to } 1.0\,r_{\text{match}}.rcluster-same​≈0.75rmatch​ to 1.0rmatch​.

From each cluster keep only the highest-score prediction.

This handles residual duplicate peaks on the same local structure even when there is no GT nearby.

Rule D: exclude inter-class conflicts.  
If a positive and a negative prediction are within

rinterclass-conflictr_{\text{interclass-conflict}}rinterclass-conflict​

of each other, do not mine either one as a clean negative. Mark them separately as `interclass_conflict`.

Reason: that region is not “the model hallucinated a cell-like background structure.” It is “both channels are firing on the same structure.” That is a decoder / class-separation issue, not clean hard background.

A good initial value is roughly your desired minimum inter-class separation. If you do not already have one, start with rinterclass-conflict=rmatchr_{\text{interclass-conflict}} = r_{\text{match}}rinterclass-conflict​=rmatch​.

Rule E: mine only confident false positives.  
Use a mining threshold

τmine≥τeval.\tau_{\text{mine}} \ge \tau_{\text{eval}}.τmine​≥τeval​.

You want errors the model genuinely believes, not low-score clutter.

A practical default is: use the same threshold that gives best validation F1, then later try a slightly stricter threshold for mining. For example, if your evaluation threshold is 0.35, test mining at 0.35 and 0.45.

Rule F: optional border exclusion.  
If a candidate is too close to the image edge for your planned crop-based inspection, either mark it or exclude it from visual audit. For detector retraining this is optional; for visualization it is useful.

That gives you a clean definition of a mined false positive:

A prediction is mined iff it is unmatched, above the mining threshold, not within rexclude-anyr_{\text{exclude-any}}rexclude-any​ of any GT, not part of an inter-class conflict, and is the highest-scoring member of its same-class local cluster.

Formally:

mine(pi)=1\text{mine}(p_i)=1mine(pi​)=1

iff all are true:

unmatched(pi),si≥τmine,dGT-any(pi)>rexclude-any,dopp-pred(pi)>rinterclass-conflict,pi=arg⁡max⁡pj∈Cisj\text{unmatched}(p_i), \quad s_i \ge \tau_{\text{mine}}, \quad d_{\text{GT-any}}(p_i) > r_{\text{exclude-any}}, \quad d_{\text{opp-pred}}(p_i) > r_{\text{interclass-conflict}}, \quad p_i = \arg\max_{p_j \in C_i} s_junmatched(pi​),si​≥τmine​,dGT-any​(pi​)>rexclude-any​,dopp-pred​(pi​)>rinterclass-conflict​,pi​=argpj​∈Ci​max​sj​

where CiC_iCi​ is its same-class local cluster.

I would also assign each rejected unmatched prediction a reason code. That will make the analysis much easier later. Use a single categorical `filter_reason`, for example:

- `matched`
- `near_gt_any`
- `sameclass_duplicate`
- `interclass_conflict`
- `below_score_threshold`
- `kept_for_mining`

That one column will immediately tell you where the false positives are coming from.

The order of operations matters. Use this order:

1. decode predictions
2. class-aware GT matching
3. score threshold
4. near-GT exclusion
5. inter-class conflict exclusion
6. same-class clustering / top-score keep
7. write final mined CSV

I would not reverse 4 and 6. Near-GT exclusion should happen before same-class deduplication because split peaks around one real cell are the major contamination source.

For the fold protocol itself, keep it simple:

`train_points.csv` → image-level 5-fold split → for each fold:

- `fold_k_train_images.txt`
- `fold_k_holdout_images.txt`
- `fold_k_checkpoint.pt`
- `fold_k_raw_predictions.csv`
- `fold_k_prediction_relations.csv`
- `fold_k_mined_fp.csv`

Then concatenate all fold mined CSVs into:

- `train_oof_mined_fp.csv`

Now the visualization step. Yes, you should absolutely do it before retraining. In your case this is not optional.

I would make a dedicated review script that reads `train_oof_mined_fp.csv` and produces two kinds of outputs.

First output: local crops around mined points.  
For each mined point, save:

- raw RGB crop centered at the mined point,
- the same crop with the predicted point marked,
- optional nearest GT markers if any are within a wider context window.

Use at least two crop sizes for review, for example 64 and 128, even if training later uses only one.

Second output: full-image context panels.  
For each sampled mined point, save a panel with:

- full 640×640 image,
- mined point highlighted,
- all GT positive points,
- all GT negative points,
- nearby other predictions if useful.

This full-image view is important because some “false positives” only make sense when you see the neighborhood.

For sampling the review set, do not just take random examples. Use stratified sampling. I would review at least:

- 50 highest-score mined positives,
- 50 highest-score mined negatives,
- 50 random mined positives,
- 50 random mined negatives,
- 25 cases near the exclusion boundary,
- 25 cases from dense regions.

That gives you around 250 examples and usually enough to see the failure modes.

I would also prepare a small manual audit CSV for those reviewed examples with columns:

- `image_id`
- `pred_id`
- `pred_class`
- `review_label`
- `review_note`

And use review labels such as:

- `clear_non_target_cell`
- `likely_annotation_miss`
- `split_peak_near_true_cell`
- `stromal_or_lymphocyte`
- `artifact_or_noise`
- `uncertain`

That manual audit will tell you whether the mined pool is suitable for negative mining.

One important recommendation: keep cross-class-near-GT cases in a separate table instead of throwing them away completely. Those are not good negative-mining targets, but they are diagnostically valuable. Make a side file like:

- `train_oof_interclass_conflicts.csv`

That file may later tell you whether the bigger problem is background confusion or positive/negative channel competition.

My recommended default hyperparameters for your first pass:

- `K = 5`
- `r_match = current evaluation radius`
- `r_exclude_any = 1.5 * r_match` to `2.0 * r_match`
- `r_cluster_same = 0.75 * r_match`
- `r_interclass_conflict = r_match`
- `tau_mine = tau_eval` initially, then test slightly stricter
- keep only top-scoring point per same-class local cluster

For your project, the most likely contamination sources are:

- split peaks around one annotated cell,
- wrong-class predictions near a real cell,
- closely adjacent nuclei where annotation is one point but model sees two peaks,
- true but unannotated tumor-like cells.

The first two should not be mined as negatives. The last one is the dangerous one, and that is exactly why visualization is necessary before retraining.

My overall recommendation is:

Build the OOF-mined pool first, but do not feed it back into training until you have visually audited a few hundred examples and estimated the composition. If the pool is mostly stromal / lymphocyte / faint irrelevant nuclei, proceed with negative mining. If a large fraction looks like plausible missed tumor annotations, use lower weights or do not use them as strict negatives.

If you want, next I’ll turn this into a concrete implementation plan for your repo: file names, function boundaries, and the exact sequence of scripts.