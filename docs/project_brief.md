# Project brief


## Problem
Development of a model based on CNN backbone + FPN > localization head for Ki-67 assesment on BCData dataset.


## Goal
Build a clean and reproducible baseline project that includes dataset handling, target generation, model definition, training, evaluation, and visualization utilities.


## Input 
BCData image patches and corresponding point annotations for Ki-67 positive and negative nuclei.


## Target output
- predicted localization heatmaps
- decoded cell center predictions
- evaluation metrics on the BCData test set
- visualizations for sanity checking targets and predictions


## Constraints: using only bcdata dataset.
- use only BCData
- start with localization-only approach
- keep code clean and easy to extend later
- prioritize correctness and reproducibility over complexity


## Evaluation
Primary metrics:
- precision
- recall
- F1 score

Evaluation split:
- BCData test set


