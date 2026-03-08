# Architecture


## Overview
The system is a localization pipeline for Ki-67 cell detection on BCData.


## Pipeline
1. Load image patch and point annotations.
2. Convert annotations into target heatmaps for positive and negative cells.
3. Pass image through backbone + FPN.
4. Predict localization heatmaps through localization head
5. Compute training loss
6. Decode predicted heatmaps into cell center coordinates
7. Match predictions to ground truth for metric calculation
8. Save logs, checkpoints, predictions, and visualizations


## Main modules


### datasets
- reading BCData samples
- parsing annotations
- applying transforms
- returning image and targets in consistent format


### target_generation
- generating positive/negative lozalization heatmaps
- preserving coordinate conventions
- handling output resolution sclaing


### models
- backbone
- FPN
- lock head
- forward output structure


### losses
- localization losses

### training:
- epoch loops
- optimizer/scheduler steps
- logging losses
- checkpoint saving


### evaluation
- heatmap decoding
- matching predictions to GT
- precision / recall / F1 calculation


### visualization
- plotting image + points
- plotting heatmaps
- overlaying predictions vs GT


## Artifacts
- checkpoints
- train/val logs
- metric visualizations


