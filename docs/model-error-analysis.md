# Model Error Analysis Guide

## Overall review
Report the chosen regression or classification metrics with the exact target definition and evaluation split. Include a baseline so that model performance has context.

## Segment-level checks
Where sample sizes permit, inspect errors by:
- Player role or playing position
- Match format
- Venue or opposition group
- Experience or historical-match-count bucket
- Low-, medium-, and high-performance ranges

## Residual inspection
For regression tasks, inspect residual distributions and predicted-versus-actual plots. Look for systematic underprediction, overprediction, and unusually large errors.

## Practical safeguards
- Do not use post-match statistics as input features.
- Avoid drawing conclusions from very small player subgroups.
- Keep the final test set untouched during model selection.
- Record the model version, feature set, and data cutoff date.

## Improvement loop
Prioritize errors that are both frequent and explainable, then test one change at a time against the same validation protocol.
